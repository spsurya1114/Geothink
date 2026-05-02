# executor.py
import numpy as np
import rasterio
import rasterio.warp as warp
from rasterio.enums import Resampling
from pathlib import Path
from schemas import GISWorkflow, GISOperation
from whitebox import WhiteboxTools

wbt = WhiteboxTools()
wbt.set_verbose_mode(False)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_DIR = Path("data")


async def execute_workflow(workflow: GISWorkflow) -> dict:
    """
    Execute every step in a validated workflow sequentially.
    Each step's outputs are stored in `context` and passed
    as inputs to the next step that needs them.
    """
    # context holds file paths produced by each step
    # so later steps can find what earlier steps created
    context = {"workflow_region": workflow.region}
    cot_log = []

    print(f"\n[Executor] Starting execution: {len(workflow.steps)} steps")

    for step in workflow.steps:
        print(f"[Executor] Step {step.step_id}: {step.operation.value}")
        cot_log.append(
            f"Step {step.step_id} ({step.operation.value}): {step.description}"
        )

        try:
            result = _dispatch(step, context)
            context.update(result)
            print(f"[Executor] Step {step.step_id} done -> {result}")

        except Exception as e:
            error_msg = f"Step {step.step_id} ({step.operation.value}) failed: {e}"
            print(f"[Executor] ERROR: {error_msg}")
            # Raise so the ReAct loop can catch and self-heal
            raise RuntimeError(error_msg)

    print(f"[Executor] All steps complete")
    outputs = {}
    for key, value in context.items():
        if isinstance(value, Path):
            outputs[key] = str(value)
        else:
            outputs[key] = value

    return {
        "status":  "success",
        "region":  workflow.region,
        "cot_log": cot_log,
        "outputs": outputs,
    }


def _dispatch(step, context: dict) -> dict:
    """
    Route each step to the correct function.
    Returns a dict of output key -> file path.
    """

    op = step.operation

    if op == GISOperation.FETCH_DEM:
        return _fetch_dem(step, context)

    elif op == GISOperation.REPROJECT:
        return _reproject(step, context)

    elif op == GISOperation.FILL_DEPRESSIONS:
        return _fill_depressions(step, context)

    elif op == GISOperation.FLOW_DIRECTION:
        return _flow_direction(step, context)

    elif op == GISOperation.FLOW_ACCUMULATION:
        return _flow_accumulation(step, context)

    elif op == GISOperation.EXTRACT_STREAMS:
        return _extract_streams(step, context)

    elif op == GISOperation.THRESHOLD_CLASSIFY:
        return _threshold_classify(step, context)

    elif op == GISOperation.VECTOR_OVERLAY:
        return _vector_overlay(step, context)

    elif op == GISOperation.EXPORT_RESULT:
        return _export_result(step, context)

    else:
        # For operations not yet implemented, skip gracefully
        print(f"[Executor] Skipping unimplemented op: {op.value}")
        return {}


# ─────────────────────────────────────────────
# Individual step implementations
# ─────────────────────────────────────────────

def _fetch_dem(step, context):
    """
    Dynamically fetches SRTM DEM tiles for the requested region using Mapzen AWS tiles
    and OSMnx to determine the bounding box.
    """
    import osmnx as ox
    import mercantile
    import requests
    from rasterio.merge import merge
    from rasterio.io import MemoryFile

    place_name = step.inputs.get("place_name") or context.get("workflow_region", "Tamil Nadu, India")
    out_path = OUTPUT_DIR / f"{place_name}_dem.tif"

    # If already downloaded for this session, reuse it
    if out_path.exists():
        print(f"[fetch_dem] DEM already exists at {out_path}")
        return {"dem_path": str(out_path)}

    print(f"[fetch_dem] Fetching bounding box for '{place_name}'...")
    try:
        gdf = ox.geocode_to_gdf(place_name)
        bounds = gdf.total_bounds # [minx, miny, maxx, maxy]
        context["boundary_gdf"] = gdf  # Cache for vector_overlay
    except Exception as e:
        print(f"[fetch_dem] Warning: Could not resolve '{place_name}' as Polygon. Trying as Point...")
        try:
            # Fallback: geocode as a point, then create a ~20km bounding box around it
            # 0.1 degree is roughly 11km at the equator.
            lat, lon = ox.geocode(place_name)
            bounds = [lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1]
            print(f"[fetch_dem] Created 20x20km bounding box around {lat}, {lon}")
        except Exception as e2:
            raise ValueError(f"Could not resolve place name '{place_name}' via OSM at all: {e2}")

    # Calculate tiles for the bounds, starting at Zoom Level 11 (suitable for city scale DEM)
    # Mapzen elevation tiles: https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{z}/{x}/{y}.tif
    z = 11
    tiles = list(mercantile.tiles(bounds[0], bounds[1], bounds[2], bounds[3], z))
    
    # Dynamic Resolution Scaling: If the region is massive, drop resolution to speed up math.
    # Keep total tiles <= 12 to ensure pipeline finishes quickly.
    while len(tiles) > 12 and z > 6:
        z -= 1
        tiles = list(mercantile.tiles(bounds[0], bounds[1], bounds[2], bounds[3], z))
        print(f"[fetch_dem] Region is massive. Dropping resolution to Zoom Level {z} ({len(tiles)} tiles)...")

    print(f"[fetch_dem] Downloading {len(tiles)} tiles at zoom {z}...")
    
    src_files_to_mosaic = []
    memory_files = []
    
    for t in tiles:
        url = f"https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{z}/{t.x}/{t.y}.tif"
        resp = requests.get(url)
        if resp.status_code == 200:
            # Load directly into memory so we don't spam disk with small tiles
            memfile = MemoryFile(resp.content)
            src = memfile.open()
            src_files_to_mosaic.append(src)
            memory_files.append(memfile)
        else:
            print(f"[fetch_dem] Warning: Failed to fetch tile {t.x}, {t.y}")

    if not src_files_to_mosaic:
        raise RuntimeError("Failed to download any DEM tiles.")

    print(f"[fetch_dem] Merging {len(src_files_to_mosaic)} tiles...")
    mosaic, out_trans = merge(src_files_to_mosaic)
    
    out_meta = src_files_to_mosaic[0].meta.copy()
    
    # Cleanup memory files
    for src in src_files_to_mosaic:
        src.close()
    for memfile in memory_files:
        memfile.close()

    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })

    # Crop the mosaic to the exact bounding box
    from rasterio.io import MemoryFile
    from rasterio.mask import mask
    from shapely.geometry import box
    from rasterio.warp import transform_geom

    with MemoryFile() as memfile:
        with memfile.open(**out_meta) as temp_src:
            temp_src.write(mosaic)
            
        with memfile.open() as src:
            # Create a Shapely box from the original lat/lon bounds (EPSG:4326)
            bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
            
            # Transform the bbox from EPSG:4326 to the DEM's CRS (e.g. EPSG:3857)
            bbox_proj = transform_geom("EPSG:4326", src.crs, bbox)
            
            # Crop the mosaic to the bounding box
            try:
                out_image, out_transform = mask(src, [bbox_proj], crop=True)
                out_meta.update({
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })
                final_data = out_image
                print(f"[fetch_dem] Successfully cropped DEM to bounding box.")
            except ValueError:
                # Fallback if mask fails (e.g. bounds don't overlap exactly)
                print(f"[fetch_dem] Warning: Could not crop DEM to bounding box.")
                final_data = mosaic

    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(final_data)

    with rasterio.open(out_path) as src:
        print(f"[fetch_dem] DEM loaded: {src.width}x{src.height} "
              f"pixels, CRS={src.crs}, bounds={src.bounds}")

    print(f"[fetch_dem] Dynamic DEM saved to {out_path}")
    return {"dem_path": str(out_path)}


def _reproject(step, context):
    """
    Reproject the DEM to the appropriate UTM Zone.
    This is essential for accurate distance/area calculations —
    lat/lon degrees are not equal in meters across the map.
    """
    import math
    src_path   = context.get("dem_path")
    target_crs = step.inputs.get("target_crs", "auto")

    if not src_path:
        raise ValueError("reproject needs dem_path in context")

    out_path = OUTPUT_DIR / "dem_reprojected.tif"

    with rasterio.open(src_path) as src:
        if target_crs == "auto":
            from rasterio.warp import transform_bounds
            # The downloaded DEM is likely in Web Mercator (meters). We need degrees.
            minx, miny, maxx, maxy = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            # Calculate UTM zone based on center longitude
            center_lon = (minx + maxx) / 2.0
            utm_zone = math.floor((center_lon + 180) / 6.0) + 1
            target_crs = f"EPSG:326{utm_zone}"
            print(f"[reproject] Auto-detected UTM Zone {utm_zone} -> {target_crs}")

        # Skip reprojection if already in target CRS
        if str(src.crs) == target_crs:
            print(f"[reproject] Already in {target_crs}, skipping")
            return {"dem_path": src_path}

        # Calculate the new transform and dimensions
        transform, width, height = warp.calculate_default_transform(
            src.crs, target_crs,
            src.width, src.height,
            *src.bounds
        )

        profile = src.profile.copy()
        profile.update(
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
            nodata=-9999
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            warp.reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_crs=src.crs,
                dst_crs=target_crs,
                resampling=Resampling.bilinear
            )

    print(f"[reproject] Reprojected to {target_crs} -> {out_path}")
    return {"dem_path": str(out_path)}


def _fill_depressions(step, context):
    """
    Fill sinks/pits in the DEM using WhiteboxTools.
    Raw DEMs have small errors where water would pool unrealistically.
    Filling them ensures flow direction is calculated correctly.
    """
    dem_path = context.get("dem_path")
    if not dem_path:
        raise ValueError("fill_depressions needs dem_path in context")

    out_path = OUTPUT_DIR / "dem_filled.tif"
    
    # WhiteboxTools FillDepressions
    wbt.fill_depressions(dem=str(Path(dem_path).resolve()), output=str(out_path.resolve()))

    print(f"[fill_depressions] Filled DEM -> {out_path}")
    return {"filled_dem_path": str(out_path)}


def _flow_direction(step, context):
    """
    Calculate D8 flow direction using WhiteboxTools.
    Each cell gets a value indicating which of its 8 neighbors water flows toward.
    """
    dem_path = context.get("filled_dem_path") or context.get("dem_path")
    if not dem_path:
        raise ValueError("flow_direction needs filled_dem_path in context")

    out_path = OUTPUT_DIR / "flow_direction.tif"

    # WhiteboxTools D8Pointer
    wbt.d8_pointer(dem=str(Path(dem_path).resolve()), output=str(out_path.resolve()))

    print(f"[flow_direction] D8 flow direction -> {out_path}")
    return {"flow_dir_path": str(out_path)}


def _flow_accumulation(step, context):
    """
    Count how many upstream cells drain into each cell using WhiteboxTools.
    High accumulation = river channel or valley bottom.
    """
    flow_dir_path = context.get("flow_dir_path")
    if not flow_dir_path:
        raise ValueError("flow_accumulation needs flow_dir_path in context")

    out_path = OUTPUT_DIR / "flow_accumulation.tif"

    # WhiteboxTools D8FlowAccumulation (takes flow direction/pointer as input)
    wbt.d8_flow_accumulation(i=str(Path(flow_dir_path).resolve()), output=str(out_path.resolve()), out_type="cells", pntr=True)

    print(f"[flow_accumulation] Flow accumulation -> {out_path}")
    return {"flow_acc_path": str(out_path)}


def _extract_streams(step, context):
    """
    Extract stream network by thresholding flow accumulation using WhiteboxTools.
    """
    flow_acc_path = context.get("flow_acc_path")
    if not flow_acc_path:
        raise ValueError("extract_streams needs flow_acc_path in context")

    threshold = step.inputs.get("threshold", 5000)
    out_path  = OUTPUT_DIR / "streams.tif"

    # WhiteboxTools ExtractStreams
    wbt.extract_streams(flow_accum=str(Path(flow_acc_path).resolve()), output=str(out_path.resolve()), threshold=threshold)

    print(f"[extract_streams] Extracted streams (threshold={threshold}) -> {out_path}")
    return {"streams_path": str(out_path)}


def _threshold_classify(step, context):
    """
    Classify every pixel into a flood risk zone.
    Modes:
      - "fluvial" (default): Uses HAND (Height Above Nearest Drainage). Good for inland rivers.
      - "coastal": Hybrid mode. Uses both HAND and absolute DEM elevation, taking the highest risk. Good for storm surge + inland rivers.
    """
    dem_path = context.get("filled_dem_path") or context.get("dem_path")
    mode = step.inputs.get("mode", "fluvial")

    if not dem_path:
        raise ValueError("threshold_classify needs dem_path in context")

    # Ensure reasonable thresholds (LLM might hallucinate huge numbers like 75m)
    low_m  = step.inputs.get("low_m", 5)
    high_m = step.inputs.get("high_m", 15)
    
    # Cap thresholds to prevent the entire map from turning red
    if low_m > 10: low_m = 5
    if high_m > 25: high_m = 15

    out_path = OUTPUT_DIR / "flood_risk.tif"
    streams_path = context.get("streams_path")

    if not streams_path:
        raise ValueError(f"{mode} threshold_classify needs streams_path in context")

    hand_path = OUTPUT_DIR / "hand.tif"

    if mode == "fluvial":
        print("[threshold_classify] Mode: Fluvial. Computing Elevation Above Stream (HAND)...")
        wbt.elevation_above_stream(dem=str(Path(dem_path).resolve()), streams=str(Path(streams_path).resolve()), output=str(hand_path.resolve()))
        
        with rasterio.open(hand_path) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata
            profile = src.profile.copy()

        risk = np.zeros_like(data, dtype=np.uint8)
        valid_mask = (data != nodata) if nodata is not None else np.ones_like(data, dtype=bool)

        risk[valid_mask & (data > high_m)] = 1
        risk[valid_mask & (data > low_m) & (data <= high_m)] = 2
        risk[valid_mask & (data <= low_m)] = 3

    elif mode == "coastal":
        print("[threshold_classify] Mode: Coastal (Hybrid). Computing HAND and absolute DEM elevation...")
        wbt.elevation_above_stream(dem=str(Path(dem_path).resolve()), streams=str(Path(streams_path).resolve()), output=str(hand_path.resolve()))
        
        with rasterio.open(hand_path) as src:
            hand_data = src.read(1).astype(np.float32)
            nodata = src.nodata
            profile = src.profile.copy()
            
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1).astype(np.float32)

        valid_mask = (hand_data != nodata) if nodata is not None else np.ones_like(hand_data, dtype=bool)

        # Fluvial Risk (HAND)
        hand_risk = np.zeros_like(hand_data, dtype=np.uint8)
        hand_risk[valid_mask & (hand_data > high_m)] = 1
        hand_risk[valid_mask & (hand_data > low_m) & (hand_data <= high_m)] = 2
        hand_risk[valid_mask & (hand_data <= low_m)] = 3

        # Coastal Risk (Absolute Elevation)
        dem_risk = np.zeros_like(dem_data, dtype=np.uint8)
        dem_risk[valid_mask & (dem_data > high_m)] = 1
        dem_risk[valid_mask & (dem_data > low_m) & (dem_data <= high_m)] = 2
        dem_risk[valid_mask & (dem_data <= low_m)] = 3

        # Combine: take the highest risk (3 > 2 > 1 > 0)
        risk = np.maximum(hand_risk, dem_risk)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    profile.update(dtype=rasterio.uint8, nodata=0)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(risk, 1)

    high   = int((risk == 3).sum())
    medium = int((risk == 2).sum())
    low    = int((risk == 1).sum())
    print(f"[threshold_classify] Risk map -> {out_path}")
    print(f"  High risk:   {high:,} cells")
    print(f"  Medium risk: {medium:,} cells")
    print(f"  Low risk:    {low:,} cells")

    return {
        "risk_raster": str(out_path),
        "mode": mode,
        "low_m": low_m,
        "high_m": high_m,
        "stats": {
            "high_risk_cells":   high,
            "medium_risk_cells": medium,
            "low_risk_cells":    low,
        }
    }

def _vector_overlay(step, context):
    """
    Overlay the flood risk raster with a vector boundary.
    Clips the risk map to the actual city/district boundary.
    Fetches the boundary dynamically using OSMnx.
    """
    import geopandas as gpd
    import osmnx as ox
    from rasterio.mask import mask
    from shapely.geometry import mapping

    risk_path  = context.get("risk_raster")
    place_name = step.inputs.get("place_name") or context.get("workflow_region", "Tamil Nadu, India")

    if not risk_path:
        raise ValueError("vector_overlay needs risk_raster in context")

    out_path = OUTPUT_DIR / f"{place_name}_flood_risk_clipped.tif"

    print(f"[vector_overlay] Fetching boundary for '{place_name}'...")
    
    # Use cached geometry from fetch_dem if available to prevent API rate limiting
    gdf = context.get("boundary_gdf")
    if gdf is None:
        try:
            print(f"[vector_overlay] No cached boundary found. Calling OSMnx...")
            gdf = ox.geocode_to_gdf(place_name)
        except Exception as e:
            print(f"[vector_overlay] Could not fetch boundary for {place_name}: {e}")
            return {"risk_raster_clipped": risk_path}

    print(f"[vector_overlay] Found boundary for {place_name}")

    # Reproject boundary to match raster CRS
    with rasterio.open(risk_path) as src:
        raster_crs = src.crs

    gdf = gdf.to_crs(raster_crs)

    # Buffer slightly so we don't clip too tight (500m buffer)
    gdf["geometry"] = gdf.geometry.buffer(500)

    # Clip the raster to the boundary
    shapes = [mapping(geom) for geom in gdf.geometry]

    with rasterio.open(risk_path) as src:
        try:
            # Explicitly enforce nodata=0 so pixels outside the polygon become 0 (transparent)
            clipped, transform = mask(src, shapes, crop=True, nodata=0)
            profile = src.profile.copy()
            profile.update(
                transform=transform,
                width=clipped.shape[2],
                height=clipped.shape[1],
                nodata=0
            )
        except ValueError as e:
            print(f"[vector_overlay] Warning: mask failed ({e}). Returning unclipped.")
            return {"risk_raster_clipped": risk_path}

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(clipped)

    print(f"[vector_overlay] Clipped risk map -> {out_path}")
    return {"risk_raster_clipped": str(out_path)}

def _export_result(step, context):
    # Prefer clipped version if available, fall back to full raster
    risk_path = (
        context.get("risk_raster_clipped") or
        context.get("risk_raster")
    )

    if not risk_path or not Path(risk_path).exists():
        raise FileNotFoundError(
            "No flood risk raster found to export."
        )

    size_kb = Path(risk_path).stat().st_size / 1024
    print(f"[export_result] Output ready: {risk_path} ({size_kb:.1f} KB)")
    return {"final_output": risk_path}