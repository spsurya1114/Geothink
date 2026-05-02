
import gradio as gr
import asyncio
import json
import threading
import queue
from pathlib import Path
from agents import GeoThinkOrchestrator

orchestrator = GeoThinkOrchestrator()


def run_geothink(query: str, region: str):
    if not query.strip():
        yield "Please enter a query.", "", None, gr.Accordion(open=True)
        return

    log_queue  = queue.Queue()
    done_event = threading.Event()
    result_box = {}

    def log_cb(msg: str):
        log_queue.put(msg)

    def run_async():
        async def _run():
            try:
                result = await orchestrator.run(
                    query        = query,
                    region       = region,
                    log_callback = log_cb
                )
                result_box["result"] = result
            except Exception as e:
                result_box["error"] = str(e)
                log_cb(f"\n[ERROR] {str(e)}")
            finally:
                done_event.set()
        asyncio.run(_run())

    threading.Thread(target=run_async, daemon=True).start()

    collected_logs = []
    while not done_event.is_set() or not log_queue.empty():
        try:
            msg = log_queue.get(timeout=0.2)
            collected_logs.append(msg)
            yield "\n".join(collected_logs), "", None, gr.Accordion(open=True)
        except queue.Empty:
            continue

    if "error" in result_box:
        yield "\n".join(collected_logs), f"**Error:** {result_box['error']}", None, gr.Accordion(open=True)
        return

    result    = result_box["result"]
    cot       = result.get("cot_log", [])
    outputs   = result.get("outputs", {})
    agents    = result.get("agents_used", [])
    reasoning = result.get("reasoning", "")

    # Get stats directly from outputs dict (most reliable)
    high = medium = low = 0
    stats_raw = result.get("outputs", {}).get("stats", {})

    if isinstance(stats_raw, dict):
        high   = stats_raw.get("high_risk_cells",   0)
        medium = stats_raw.get("medium_risk_cells",  0)
        low    = stats_raw.get("low_risk_cells",     0)
    else:
        # Fallback: parse from cot_log
        for line in cot:
            if "High risk:" in line:
                try:
                    high = int(line.split(":")[1].strip().split()[0].replace(",",""))
                except Exception:
                    pass
            if "Medium risk:" in line:
                try:
                    medium = int(line.split(":")[1].strip().split()[0].replace(",",""))
                except Exception:
                    pass
            if "Low risk:" in line:
                try:
                    low = int(line.split(":")[1].strip().split()[0].replace(",",""))
                except Exception:
                    pass

    total      = high + medium + low or 1
    stats_text = f"""
## Results for {region}

**Agents:** {' -> '.join(agents)}

**Reasoning:** {reasoning}

### Flood Risk Classification
| Risk Level | Cells | Percentage |
|------------|-------|------------|
| 🔴 High    | {high:,} | {100*high/total:.1f}% |
| 🟡 Medium  | {medium:,} | {100*medium/total:.1f}% |
| 🟢 Low     | {low:,} | {100*low/total:.1f}% |

### Output Files
"""
    for key, path in outputs.items():
        if key != "stats":
            stats_text += f"- `{key}`: `{path}`\n"

    image_path = _generate_map(region, outputs)
    yield "\n".join(collected_logs), stats_text, image_path, gr.Accordion(open=False)

def _generate_map(region: str, outputs: dict) -> str:
    """Generate an interactive Folium map from the flood risk tif."""
    import numpy as np
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.transform import array_bounds
    import folium
    from folium.plugins import Geocoder
    import base64
    from io import BytesIO
    from PIL import Image
    from pathlib import Path

    # Try clipped first, then full raster
    risk_path_str = outputs.get("risk_raster_clipped") or outputs.get("risk_raster") or outputs.get("final_output")
    if not risk_path_str:
        risk_path = Path(f"outputs/{region}_flood_risk_clipped.tif")
        if not risk_path.exists():
            risk_path = Path("outputs/flood_risk_clipped.tif")
    else:
        risk_path = Path(risk_path_str)

    if not risk_path.exists():
        return None

    with rasterio.open(risk_path) as src:
        dst_crs = 'EPSG:4326'
        # Calculate transform and dimensions in EPSG:4326
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        
        out_image = np.zeros((1, height, width), dtype=src.dtypes[0])
        reproject(
            source=rasterio.band(src, 1),
            destination=out_image,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest)
        
        # calculate bounds in EPSG:4326
        bounds = array_bounds(height, width, transform)
        # bounds are (left, bottom, right, top) in lon/lat
        folium_bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

    risk_map = out_image[0]
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[risk_map == 1] = [46, 204, 113, 200]   # Low risk (Green)
    rgba[risk_map == 2] = [243, 156, 18, 200]   # Medium risk (Orange)
    rgba[risk_map == 3] = [231, 76, 60, 200]    # High risk (Red)

    img = Image.fromarray(rgba)
    img_data = BytesIO()
    img.save(img_data, format='PNG')
    b64_img = base64.b64encode(img_data.getvalue()).decode()
    img_url = f'data:image/png;base64,{b64_img}'

    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    # Create Folium Map with OpenStreetMap for clear labels
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='OpenStreetMap')
    
    # Add Image Overlay
    folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=folium_bounds,
        opacity=0.6,
        interactive=True,
        cross_origin=False,
    ).add_to(m)

    # Add Geocoder search bar
    Geocoder(position='topright').add_to(m)

    # Add HTML Legend
    legend_html = '''
    <div style="position: fixed; 
        bottom: 30px; left: 30px; width: 230px; height: 130px; 
        background-color: rgba(20, 20, 30, 0.85); backdrop-filter: blur(10px);
        z-index:9999; font-size:14px; border:1px solid rgba(255,255,255,0.1);
        border-radius: 12px; color: white; padding: 15px; font-family: sans-serif;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);">
        <b>Flood Risk Levels</b><br><br>
        <i style="background:#e74c3c;width:15px;height:15px;float:left;margin-right:8px;border-radius:3px;"></i> High risk (HAND < low_m)<br>
        <i style="background:#f39c12;width:15px;height:15px;float:left;margin-right:8px;border-radius:3px;"></i> Medium risk (low_m - high_m)<br>
        <i style="background:#2ecc71;width:15px;height:15px;float:left;margin-right:8px;border-radius:3px;"></i> Low risk (HAND > high_m)<br>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    return m._repr_html_()


def create_ui():
    custom_css = """
        body { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); }
        .gradio-container { 
            max-width: 95% !important; 
            margin: auto; 
            background: rgba(255, 255, 255, 0.05) !important; 
            backdrop-filter: blur(15px); 
            -webkit-backdrop-filter: blur(15px);
            border-radius: 20px; 
            border: 1px solid rgba(255, 255, 255, 0.1); 
            padding: 30px;
            box-shadow: 0 25px 45px rgba(0, 0, 0, 0.2);
        }
        .log-box { 
            height: 500px !important; 
            overflow-y: auto !important; 
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .log-box textarea { 
            height: 500px !important; 
            overflow-y: auto !important;
            font-family: 'Consolas', monospace;
            font-size: 14px;
            line-height: 1.6;
            background-color: rgba(0, 0, 0, 0.6) !important; 
            color: #00ffcc !important;
            border: none;
        }
        h1, p, span { color: #ffffff !important; }
        .map-container { 
            height: 600px !important; 
            border-radius: 12px; 
            overflow: hidden; 
            border: 1px solid rgba(255, 255, 255, 0.1); 
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            margin-bottom: 20px;
        }
        .map-container iframe {
            border: none;
            width: 100%;
            height: 100%;
        }
        .gradio-button.primary {
            background: linear-gradient(90deg, #00c6ff, #0072ff) !important;
            border: none !important;
            box-shadow: 0 4px 15px rgba(0, 114, 255, 0.4) !important;
            transition: all 0.3s ease;
        }
        .gradio-button.primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 114, 255, 0.6) !important;
        }
    """
    with gr.Blocks(
        title="GeoThink — Flood Risk AI",
        theme=gr.themes.Base(),
        css=custom_css
    ) as demo:

        # Header
        gr.Markdown("""
        # 🌊 GeoThink — Flood Risk Modeling
        **Reasoning-Enabled AI Agents for Natural Language-Driven GIS Analysis**
        
        *SASTRA Deemed University — School of Computing — CSE (AI & DS) 2023–27*
        """)

        # Wrap everything in Tabs
        with gr.Tabs():
            # TAB 1: Analysis Pipeline
            with gr.Tab("Analysis Pipeline"):
                # Input row
                with gr.Row():
                    with gr.Column(scale=3):
                        query_input = gr.Textbox(
                            label       = "Natural Language Query",
                            placeholder = "Show flood risk for Trichy using terrain analysis",
                            lines       = 2,
                        )
                    with gr.Column(scale=1):
                        region_input = gr.Textbox(
                            label   = "Region",
                            value   = "Trichy",
                            lines   = 2,
                        )

                run_btn = gr.Button(
                    "▶ Run Flood Risk Analysis",
                    variant="primary",
                    size="lg"
                )

                map_output = gr.HTML(
                    label = "Interactive Flood Risk Map",
                    elem_classes=["map-container"]
                )

                with gr.Accordion("Agent Pipeline Logs", open=True) as log_accordion:
                    log_output = gr.Textbox(
                        label    = "Execution Logs",
                        lines    = 15, 
                        elem_classes = ["log-box"],
                        interactive  = False,
                        autoscroll   = True,
                        show_label   = False
                    )

                stats_output = gr.Markdown(label="Results")

                gr.Examples(
                    examples=[
                        ["Show flood risk for Trichy using terrain analysis", "Trichy"],
                        ["Identify high risk flood zones in Tiruchirappalli", "Trichy"],
                        ["Analyze flood prone areas near Cauvery river", "Trichy"],
                    ],
                    inputs=[query_input, region_input],
                    label="Example Queries"
                )

                run_btn.click(
                    fn      = run_geothink,
                    inputs  = [query_input, region_input],
                    outputs = [log_output, stats_output, map_output, log_accordion],
                )

            # TAB 2: Model Evaluation
            with gr.Tab("Model Evaluation"):
                gr.Markdown("### 📊 Ground-Truth Evaluation Dashboard")
                gr.Markdown("Upload a historical flood Shapefile (.zip or .shp) to mathematically compare against our AI's prediction.")
                
                with gr.Row():
                    with gr.Column():
                        import glob
                        import os
                        def get_prediction_files():
                            files = glob.glob("outputs/*_flood_risk_clipped.tif")
                            return [os.path.basename(f) for f in files]
                            
                        pred_dropdown = gr.Dropdown(
                            choices=get_prediction_files(), 
                            label="Select AI Prediction Raster", 
                            info="Select a previously generated flood risk map."
                        )
                        refresh_btn = gr.Button("↻ Refresh List", size="sm")
                        
                        gt_upload = gr.File(
                            label="Upload Ground Truth Shapefile (.zip or .shp)",
                            file_types=[".zip", ".shp"]
                        )
                        
                        eval_btn = gr.Button("📊 Run Evaluation", variant="primary")
                        
                    with gr.Column():
                        eval_metrics = gr.Markdown("### Evaluation Metrics\nRun evaluation to see accuracy scores.")
                        eval_img = gr.HTML(label="Visualization Map")

                def run_evaluation(pred_filename, gt_file):
                    if not pred_filename or not gt_file:
                        return "Please select both a prediction file and a ground truth file.", ""
                    
                    try:
                        from evaluate import evaluate_flood_model
                        pred_path = f"outputs/{pred_filename}"
                        gt_path = gt_file.name
                        
                        metrics, img_b64 = evaluate_flood_model(pred_path, gt_path)
                        
                        md = f"""
                        ### Results
                        - **Accuracy:** `{metrics['accuracy']}%`
                        - **Precision:** `{metrics['precision']}%`
                        - **Recall:** `{metrics['recall']}%`
                        - **IoU (Intersection over Union):** `{metrics['iou']}%`
                        """
                        
                        img_html = f'''
                        <div style="border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; overflow: hidden; background: #000;">
                            <img src="data:image/png;base64,{img_b64}" style="width: 100%; height: auto; display: block;" />
                            <div style="padding: 10px; background: rgba(0,0,0,0.8); color: white; font-family: sans-serif; font-size: 14px;">
                                <b>Legend:</b> 
                                <span style="color:#2ecc71; margin-right:15px;">■ True Positive (Correct)</span>
                                <span style="color:#e74c3c; margin-right:15px;">■ False Positive (False Alarm)</span>
                                <span style="color:#3498db;">■ False Negative (Missed Flood)</span>
                            </div>
                        </div>
                        '''
                        return md, img_html
                    except Exception as e:
                        return f"### Error during evaluation\n`{str(e)}`", ""

                eval_btn.click(
                    fn=run_evaluation,
                    inputs=[pred_dropdown, gt_upload],
                    outputs=[eval_metrics, eval_img]
                )
                
                refresh_btn.click(
                    fn=lambda: gr.update(choices=get_prediction_files()),
                    inputs=[],
                    outputs=[pred_dropdown]
                )

    return demo

if __name__ == "__main__":
    ui = create_ui()
    ui.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        share       = False,
        inbrowser   = True,
    )