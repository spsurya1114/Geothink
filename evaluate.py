import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.warp import calculate_default_transform, reproject, Resampling
from PIL import Image
import io
import base64

def evaluate_flood_model(prediction_tif: str, ground_truth_file: str):
    """
    Evaluates a predicted flood risk raster against a ground-truth shapefile/zip.
    Returns metrics (Accuracy, Precision, Recall, IoU) and a base64 comparison image.
    """
    print(f"[Evaluate] Loading prediction: {prediction_tif}")
    
    # 1. Load Prediction Raster
    with rasterio.open(prediction_tif) as src:
        pred_data = src.read(1)
        pred_transform = src.transform
        pred_crs = src.crs
        pred_shape = pred_data.shape
        pred_nodata = src.nodata

    # Create Binary Prediction Mask (High Risk (3), Medium (2) are considered flooded for evaluation)
    # We ignore Low Risk (1) and NoData (0)
    pred_mask = np.zeros_like(pred_data, dtype=np.uint8)
    pred_mask[(pred_data == 2) | (pred_data == 3)] = 1
    
    # Track valid area (ignore areas outside the city boundary)
    if pred_nodata is not None:
        valid_area_mask = (pred_data != pred_nodata) & (pred_data != 0)
    else:
        valid_area_mask = (pred_data != 0)

    # 2. Load Ground Truth Shapefile
    print(f"[Evaluate] Loading ground truth: {ground_truth_file}")
    if ground_truth_file.endswith('.zip'):
        # geopandas can read directly from zip
        gdf = gpd.read_file(f"zip://{ground_truth_file}")
    else:
        gdf = gpd.read_file(ground_truth_file)

    print(f"[Evaluate] Reprojecting ground truth to match prediction...")
    gdf = gdf.to_crs(pred_crs)

    # 3. Rasterize Ground Truth to match prediction pixel grid
    print(f"[Evaluate] Rasterizing ground truth polygons...")
    shapes = ((geom, 1) for geom in gdf.geometry)
    
    gt_mask = rasterize(
        shapes=shapes,
        out_shape=pred_shape,
        transform=pred_transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8
    )

    # 4. Calculate Metrics (Only within the valid city boundary)
    print(f"[Evaluate] Calculating metrics...")
    
    pred_valid = pred_mask[valid_area_mask]
    gt_valid   = gt_mask[valid_area_mask]

    tp = np.sum((pred_valid == 1) & (gt_valid == 1))
    fp = np.sum((pred_valid == 1) & (gt_valid == 0))
    fn = np.sum((pred_valid == 0) & (gt_valid == 1))
    tn = np.sum((pred_valid == 0) & (gt_valid == 0))

    accuracy  = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    # 5. Generate Visualization Map
    print(f"[Evaluate] Generating visualization map...")
    
    # RGBA image (initialized to transparent)
    vis = np.zeros((pred_shape[0], pred_shape[1], 4), dtype=np.uint8)
    
    # Identify indices where there is valid data
    rows = np.any(valid_area_mask, axis=1)
    cols = np.any(valid_area_mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        print("[Evaluate] Error: No valid data found in prediction raster.")
        return metrics, ""

    # --- COLOR LOGIC ---
    # We want a "Difference Map" look:
    
    # 1. True Positives (Green) - AI was RIGHT
    vis[(pred_mask == 1) & (gt_mask == 1)] = [46, 204, 113, 255]
    
    # 2. False Positives (Red) - AI was PESSIMISTIC (Predicted risk but no actual flood)
    # Make this semi-transparent so it's not a giant solid blob
    vis[(pred_mask == 1) & (gt_mask == 0)] = [231, 76, 60, 160]
    
    # 3. False Negatives (Blue) - AI was WRONG (Missed a flood)
    vis[(pred_mask == 0) & (gt_mask == 1) & valid_area_mask] = [52, 152, 219, 255]

    # Crop visualization to the actual data extent
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    vis_cropped = vis[rmin:rmax+1, cmin:cmax+1]

    # Convert to base64 for Gradio HTML rendering
    img = Image.fromarray(vis_cropped)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    metrics = {
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "iou": round(iou * 100, 2)
    }

    print(f"[Evaluate] Completed. Shape: {vis_cropped.shape}, IoU: {metrics['iou']}%")
    return metrics, img_str
