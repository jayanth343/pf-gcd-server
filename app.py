import os
import tempfile
import numpy as np
import asyncio
import cv2
from pydantic import BaseModel
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from typing import Any, Dict, List, Tuple, Union
from supabase import create_client, Client
import heapq
from math import radians, sin, cos, sqrt, atan2
import requests
from utils import load_rgb, extract_exif
from dotenv import load_dotenv
import os
import json
from contextlib import asynccontextmanager


load_dotenv()
device = 'cuda'

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")

supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

SNAP_THRESHOLD = 3.0



@asynccontextmanager
async def lifespan(app: FastAPI):
    depth_model, image_processor = initialize_depth_pipeline()
    depth_estimator = DepthEstimator(depth_model, image_processor)

    NavigationGraph.load_graph()
    print("graph status: ", NavigationGraph.graph==None)
    yield
    print("Done")

async def preload_models():
    global depth_estimator
    depth_model, image_processor = initialize_depth_pipeline()
    depth_estimator = DepthEstimator(depth_model, image_processor)
    NavigationGraph.load_graph()
    print("✅ Model + Graph Loaded (Background)")

app = FastAPI(
    title="Depth Pro Distance Estimation", 
    description="Estimate distance and depth using Apple's Depth Pro model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # allow all origins
    allow_credentials=True,     # allow cookies/auth headers
    allow_methods=["*"],        # allow all HTTP methods
    allow_headers=["*"],        # allow all headers
)

def initialize_depth_pipeline():
    """Initialize the Depth Pro pipeline"""
    try:
        print("Initializing Depth Pro pipeline...")
        image_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
        model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device)

        return model, image_processor
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("Falling back to dummy pipeline...")
        return None


class DepthEstimator:
    def __init__(self, model, image_processor):
        self.device = torch.device('cuda') 
        print("Initializing Depth Pro estimator...")
        self.model = model
        self.image_processor = image_processor
        print("Depth Pro estimator initialized successfully!")

    def estimate_depth(self, image_path):
        try:
            image = Image.open(image_path)
            
            resized_image, new_size = self.resize_image(image_path)

            rgb_image = load_rgb(resized_image.name)
            f_px = rgb_image[-1]
            eval_image = rgb_image[0]

            inputs = self.image_processor(eval_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            post_processed_output = self.image_processor.post_process_depth_estimation(
                outputs, target_sizes=[(new_size[1], new_size[0])],
            )
            result = post_processed_output[0]
            field_of_view = result["field_of_view"]
            focal_length = result["focal_length"] 
            depth = result["predicted_depth"]

            if isinstance(depth, torch.Tensor):
                depth = depth.detach().cpu().numpy()
            elif not isinstance(depth, np.ndarray):
                depth = np.array(depth)
            
            print(f_px,focal_length)

            
            return depth, new_size, focal_length

        except Exception as e:
            print(f"Error in depth estimation: {e}")
            return None, None, None
    
    def resize_image(self, image_path, max_size=1536):
        with Image.open(image_path) as img:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                img.save(temp_file, format="PNG")
                return temp_file, new_size
    

def find_topmost_pixel(mask):
    '''Top Pixel from footpath mask'''
    footpath_pixels = np.where(mask > 0)
    if len(footpath_pixels[0]) == 0:
        return None
    min_y = np.min(footpath_pixels[0])
    top_pixels_mask = footpath_pixels[0] == min_y
    top_x_coords = footpath_pixels[1][top_pixels_mask]
    center_idx = len(top_x_coords) // 2
    return (min_y, top_x_coords[center_idx])

def find_bottommost_footpath_pixel(mask, topmost_pixel):
    """Find the bottommost pixel perpendicular to the topmost pixel within the mask"""
    if topmost_pixel is None:
        return None
    
    top_y, top_x = topmost_pixel
    
    # Find all mask pixels in the same x-column as the topmost pixel
    mask_y_coords, mask_x_coords = np.where(mask > 0)
    column_mask = mask_x_coords == top_x
    column_y_coords = mask_y_coords[column_mask]
    
    if len(column_y_coords) == 0:
        # If no pixels in the same column, find the bottommost pixel in the entire mask
        footpath_pixels = np.where(mask > 0)
        if len(footpath_pixels[0]) == 0:
            return None
        max_y = np.max(footpath_pixels[0])
        bottom_pixels_mask = footpath_pixels[0] == max_y
        bottom_x_coords = footpath_pixels[1][bottom_pixels_mask]
        center_idx = len(bottom_x_coords) // 2
        return (max_y, bottom_x_coords[center_idx])
    
    # Find the bottommost pixel in the same x-column
    max_y_in_column = np.max(column_y_coords)
    return (max_y_in_column, top_x)


def estimate_real_world_distance(depth_map, topmost_pixel, mask):
    """Estimate real-world distance between two pixels using depth information"""

    if topmost_pixel is None or depth_map is None:
        return None
    
    bottommost_pixel = find_bottommost_footpath_pixel(mask, topmost_pixel)
    
    if bottommost_pixel is None:
        return None
    
    top_y, top_x = topmost_pixel
    bottom_y, bottom_x = bottommost_pixel
    
    if (top_y >= depth_map.shape[0] or top_x >= depth_map.shape[1] or
        bottom_y >= depth_map.shape[0] or bottom_x >= depth_map.shape[1]):
        return None
    
    topmost_depth = depth_map[top_y, top_x]
    bottommost_depth = depth_map[bottom_y, bottom_x]
    
    if np.isnan(topmost_depth) or np.isnan(bottommost_depth):
        print("Invalid depth values (NaN) found")
        return None
    
    distance_meters = float(topmost_depth - bottommost_depth)
    
    print(f"Distance calculation:")
    print(f"  Topmost pixel: ({top_y}, {top_x}) = {topmost_depth:.3f}m")
    print(f"  Bottommost pixel: ({bottom_y}, {bottom_x}) = {bottommost_depth:.3f}m")
    print(f"  Distance: {distance_meters:.3f}m")
    print(f"  Depth map shape: {depth_map.shape}")
    print(f"  Depth map dtype: {depth_map.dtype}")
    print(f"  Depth range: {np.min(depth_map):.3f}m to {np.max(depth_map):.3f}m")
    
    return distance_meters


class CoordinateSnapper:
    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371e3  # meters
        φ1 = radians(lat1)
        φ2 = radians(lat2)
        dφ = radians(lat2 - lat1)
        dλ = radians(lon2 - lon1)

        a = sin(dφ/2)**2 + cos(φ1) * cos(φ2) * sin(dλ/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    @staticmethod
    def find_nearest_node(target_lat, target_lon, existing_nodes, threshold=SNAP_THRESHOLD):
        min_distance = float('inf')
        nearest_node = None

        for node in existing_nodes:
            node_lat, node_lon = node
            dist = CoordinateSnapper.haversine_distance(target_lat, target_lon, node_lat, node_lon)

            if dist < min_distance:
                min_distance = dist
                nearest_node = node

        if nearest_node and min_distance <= threshold:
            return (*nearest_node, min_distance, True)

        return (target_lat, target_lon, min_distance, False)

    @staticmethod
    def snap_coordinates(start_lat, start_lon, end_lat, end_lon, graph_nodes):
        existing_nodes = list(graph_nodes.keys()) if graph_nodes else []

        s_lat, s_lon, s_dist, s_snap = CoordinateSnapper.find_nearest_node(
            start_lat, start_lon, existing_nodes
        )
        e_lat, e_lon, e_dist, e_snap = CoordinateSnapper.find_nearest_node(
            end_lat, end_lon, existing_nodes
        )

        info = {
            "start_snapped": s_snap,
            "start_distance": s_dist,
            "start_original": (start_lat, start_lon),
            "start_snapped_to": (s_lat, s_lon),
            "end_snapped": e_snap,
            "end_distance": e_dist,
            "end_original": (end_lat, end_lon),
            "end_snapped_to": (e_lat, e_lon),
        }

        return (s_lat, s_lon), (e_lat, e_lon), info


class Navigator:

    @staticmethod
    def haversine(a, b):
        R = 6371  # km
        lat1, lon1 = a
        lat2, lon2 = b
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        x = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
        return R * 2 * atan2(sqrt(x), sqrt(1-x)) * 1000

    def astar(self, graph, start, goal):
        open_set = [(0, start)]
        came_from = {}
        g = {start: 0}
        explored = 0

        while open_set:
            _, curr = heapq.heappop(open_set)
            explored += 1

            if curr == goal:
                path = [curr]
                while curr in came_from:
                    curr = came_from[curr]
                    path.append(curr)
                return path[::-1]

            for neigh, cost in graph.get(curr, []):
                temp_g = g[curr] + cost
                if temp_g < g.get(neigh, float('inf')):
                    g[neigh] = temp_g
                    f = temp_g + self.haversine(neigh, goal)
                    heapq.heappush(open_set, (f, neigh))
                    came_from[neigh] = curr

        return None


class NavigationGraph:
    graph: Dict = None

    @classmethod
    def load_graph(cls):
        print("\n===== LOADING GRAPH =====")
        rows = supabase.table("location-footpath").select("*").execute().data
        cls.graph = {}
        edge_count = 0

        for row in rows:
            start = (row["latitude"], row["longitude"])
            end = (row["latitude_end"], row["longitude_end"])
            score = max(row.get("score", 50), 1)

            distance = Navigator.haversine(start, end)
            cost = distance / ((score/100) ** 2)

            cls.graph.setdefault(start, []).append((end, cost))
            cls.graph.setdefault(end, []).append((start, cost))
            edge_count += 2

        print("Graph loaded:", len(cls.graph), "nodes", edge_count, "edges")

class SnapRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float


class NavigateRequest(BaseModel):
    start: List[float]
    goal: List[float]



@app.get("/")
def root():
    return {"status": "alive"}


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Depth Pro Distance Estimation API",
        "docs": "/docs",
        "health": "/health",
        "estimate_endpoint": "/estimate-depth"
    }

@app.post("/estimate-depth")
async def estimate_depth_endpoint(file: UploadFile = File(...), mask: UploadFile = File(...)):
    """FastAPI endpoint for depth estimation and distance calculation"""
    try:
        

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as mtemp_file:
            content = await mask.read()
            mtemp_file.write(content)
            temp_file_path_mask = mtemp_file.name

        # Load image for pixel detection
        image = cv2.imread(temp_file_path)
        mask = cv2.imread(temp_file_path_mask)
        if image is None or mask is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not load image or mask"}
            )
        
        # Estimate depth
        depth_map, new_size, focal_length_px = depth_estimator.estimate_depth(temp_file_path)
        
        if depth_map is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Depth estimation failed"}
            )
        
        # Resize image and mask to match depth map size
        resized_image = cv2.resize(image, new_size)
        resized_mask = cv2.resize(mask, new_size)
        
        # Convert mask to grayscale if it's not already
        if len(resized_mask.shape) == 3:
            resized_mask = cv2.cvtColor(resized_mask, cv2.COLOR_BGR2GRAY)
        
        # Find key pixels from the mask
        topmost_pixel = find_topmost_pixel(resized_mask)
        
        # Calculate distance
        distance_meters = estimate_real_world_distance(depth_map, topmost_pixel, resized_mask)
        
        # Clean up
        os.unlink(temp_file_path)
        os.unlink(temp_file_path_mask)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        result = {
            "depth_map_shape": depth_map.shape,
            "focal_length_px": float(focal_length_px) if focal_length_px is not None else None,
            "topmost_pixel": [ int(topmost_pixel[0]), int(topmost_pixel[1])] if topmost_pixel else None,
            "distance_meters": distance_meters,
            "depth_stats": {
                "min_depth": float(np.min(depth_map)),
                "max_depth": float(np.max(depth_map)),
                "mean_depth": float(np.mean(depth_map))
            }
        }
        
        return JSONResponse(content=result)
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={"error": "Request timed out"}
        )
    except Exception as e:
        # Clean up on error
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        if 'temp_file_path_mask' in locals():
            try:
                os.unlink(temp_file_path_mask)
            except:
                pass
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/frontend", response_class=HTMLResponse)
async def frontend():
    """Root endpoint with simple HTML interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Depth Pro Distance Estimation</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 800px; 
                margin: 0 auto; 
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 { 
                color: #2c3e50; 
                text-align: center;
                margin-bottom: 10px;
            }
            .subtitle {
                text-align: center;
                color: #7f8c8d;
                margin-bottom: 30px;
            }
            .upload-section {
                border: 2px dashed #3498db;
                border-radius: 10px;
                padding: 30px;
                text-align: center;
                margin: 20px 0;
                background-color: #ecf0f1;
            }
            input[type="file"] {
                margin: 10px 0;
                padding: 10px;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
            }
            .file-group {
                margin: 20px 0;
            }
            .file-label {
                display: block;
                margin-bottom: 8px;
                font-weight: bold;
                color: #2c3e50;
            }
            button {
                background-color: #3498db;
                color: white;
                padding: 12px 25px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #2980b9;
            }
            .results {
                margin-top: 20px;
                padding: 20px;
                border-radius: 5px;
                background-color: #e8f5e8;
                display: none;
            }
            .error {
                background-color: #ffeaa7;
                border-left: 4px solid #fdcb6e;
                padding: 10px;
                margin: 10px 0;
            }
            .endpoint-info {
                background-color: #74b9ff;
                color: white;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }
            .feature {
                margin: 10px 0;
                padding: 10px;
                border-left: 3px solid #3498db;
                background-color: #f8f9fa;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Depth Pro Distance Estimation</h1>
            <p class="subtitle">Upload an image and a footpath mask to estimate depth and calculate distances using Apple's Depth Pro model</p>
            
            <div class="upload-section">
                <h3>Upload Image and Mask</h3>
                <form id="uploadForm" enctype="multipart/form-data">
                    <div style="margin: 20px 0;">
                        <label for="imageFile" style="display: block; margin-bottom: 5px; font-weight: bold;">📸 Main Image:</label>
                        <input type="file" id="imageFile" name="file" accept="image/*" required style="width: 100%;">
                    </div>
                    <div style="margin: 20px 0;">
                        <label for="maskFile" style="display: block; margin-bottom: 5px; font-weight: bold;">🎭 Footpath Mask:</label>
                        <input type="file" id="maskFile" name="mask" accept="image/*" required style="width: 100%;">
                    </div>
                    <button type="submit">Analyze Image with Mask</button>
                </form>
                
                <div id="results" class="results">
                    <h3>Analysis Results:</h3>
                    <div id="resultsContent"></div>
                </div>
            </div>
            
            <div class="endpoint-info">
                <h3>🔗 API Endpoints</h3>
                <p><strong>POST /estimate-depth</strong> - Upload image and footpath mask for depth estimation</p>
                <p><strong>GET /docs</strong> - API documentation</p>
                <p><strong>GET /health</strong> - Health check</p>
            </div>
            
            <div class="feature">
                <h3>✨ Features</h3>
                <ul>
                    <li>🎯 Monocular depth estimation using Depth Pro</li>
                    <li>🎭 Footpath mask-based analysis</li>
                    <li>📏 Real-world distance calculation between mask boundaries</li>
                    <li>🖥️ CPU-optimized processing</li>
                    <li>🚀 Fast inference suitable for real-time use</li>
                </ul>
            </div>
        </div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const fileInput = document.getElementById('imageFile');
                const maskInput = document.getElementById('maskFile');
                const resultsDiv = document.getElementById('results');
                const resultsContent = document.getElementById('resultsContent');
                
                if (!fileInput.files[0]) {
                    alert('Please select a main image file');
                    return;
                }
                
                if (!maskInput.files[0]) {
                    alert('Please select a footpath mask file');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                formData.append('mask', maskInput.files[0]);
                
                try {
                    resultsContent.innerHTML = '<p>🔄 Processing image and mask...</p>';
                    resultsDiv.style.display = 'block';
                    
                    const response = await fetch('/estimate-depth', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        
                        let html = '<h4>📊 Results:</h4>';
                        html += `<p><strong>📐 Distance:</strong> ${result.distance_meters ? result.distance_meters.toFixed(3) + ' meters' : 'N/A'}</p>`;
                        html += `<p><strong>🎯 Focal Length:</strong> ${result.focal_length_px ? result.focal_length_px.toFixed(2) + ' pixels' : 'N/A'}</p>`;
                        html += `<p><strong>📊 Depth Map Shape:</strong> ${result.depth_map_shape ? result.depth_map_shape.join(' x ') : 'N/A'}</p>`;
                        html += `<p><strong>🔝 Top Mask Pixel:</strong> ${result.topmost_pixel ? `(${result.topmost_pixel[0]}, ${result.topmost_pixel[1]})` : 'N/A'}</p>`;
                        
                        if (result.depth_stats) {
                            html += '<h4>📈 Depth Statistics:</h4>';
                            html += `<p><strong>Min Depth:</strong> ${result.depth_stats.min_depth.toFixed(3)}m</p>`;
                            html += `<p><strong>Max Depth:</strong> ${result.depth_stats.max_depth.toFixed(3)}m</p>`;
                            html += `<p><strong>Mean Depth:</strong> ${result.depth_stats.mean_depth.toFixed(3)}m</p>`;
                        }
                        
                        resultsContent.innerHTML = html;
                    } else {
                        const error = await response.json();
                        resultsContent.innerHTML = `<div class="error">❌ Error: ${error.error || 'Processing failed'}</div>`;
                    }
                } catch (error) {
                    resultsContent.innerHTML = `<div class="error">❌ Network error: ${error.message}</div>`;
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/reload-graph")
def navget():
        print("\n" + "="*60)
        print("🔄 GRAPH RELOAD REQUESTED")
        print("="*60)
        try:
            NavigationGraph.graph = None  # Force reload
            NavigationGraph.load_graph()
            return JSONResponse(
                status_code=200,
                content={
                "success": True,
                "total_nodes": len(NavigationGraph.graph) if NavigationGraph.graph is not None else 0,
                "message": "Graph reloaded successfully"
            }
            )
        except Exception as e:
            print(f"❌ Error reloading graph: {e}")
            return JSONResponse(content={
                "success": False,
                "error": str(e)
            }, status_code = 500)

@app.post("/snap-coordinates")
def snap(start_lat: float, start_lon : float, end_lat : float, end_lon: float):
    if not all([start_lat, start_lon, end_lat, end_lon]):
        return JSONResponse(status_code = 400, content={"error": "All coordinates must be provided"})

    print("\n" + "="*60)
    print("🧲 COORDINATE SNAPPING REQUEST")
    print("="*60)
    print(f"📍 Original Start: {start_lat:.6f}, {start_lon:.6f}")
    print(f"📍 Original End:   {end_lat:.6f}, {end_lon:.6f}")
    print(f"📏 Snap Threshold: {SNAP_THRESHOLD}m")
    print("-"*60)

    # Perform snapping
    snapped_start, snapped_end, snap_info = CoordinateSnapper.snap_coordinates(
        start_lat, start_lon, end_lat, end_lon, NavigationGraph.graph
    )
    if snap_info['start_snapped']:
            print(f"✅ Start SNAPPED: {snap_info['start_distance']:.2f}m to existing node")
            print(f"   {snap_info['start_original']} → {snap_info['start_snapped_to']}")
    else:
        print(f"ℹ️  Start NOT snapped (nearest: {snap_info['start_distance']:.2f}m)")

    if snap_info['end_snapped']:
        print(f"✅ End SNAPPED: {snap_info['end_distance']:.2f}m to existing node")
        print(f"   {snap_info['end_original']} → {snap_info['end_snapped_to']}")
    else:
        print(f"ℹ️  End NOT snapped (nearest: {snap_info['end_distance']:.2f}m)")

    # Check if a footpath already exists with these exact coordinates
    existing_footpath = None
    try:
        response = supabase.table("location-footpath").select("*").execute()
        footpaths = response.data if response.data else []

        for fp in footpaths:
            # Check if both start and end match (with small tolerance for floating point comparison)
            start_match = (abs(fp['latitude'] - snapped_start[0]) < 0.0000001 and 
                            abs(fp['longitude'] - snapped_start[1]) < 0.0000001)
            end_match = (abs(fp['latitude_end'] - snapped_end[0]) < 0.0000001 and 
                        abs(fp['longitude_end'] - snapped_end[1]) < 0.0000001)

            if start_match and end_match:
                existing_footpath = fp
                print(f"🔍 Found existing footpath: FID={fp['fid']}")
                break
    except Exception as e:
        print(f"⚠️  Error checking for existing footpath: {e}")

    print("="*60 + "\n")
    return JSONResponse(status_code=200,content={
            "snapped_start": {
                "latitude": snapped_start[0],
                "longitude": snapped_start[1]
            },
            "snapped_end": {
                "latitude": snapped_end[0],
                "longitude": snapped_end[1]
            },
            "snap_info": snap_info,
            "existing_footpath": existing_footpath
        })

@app.post("/navigate")
def navigate(start: List[float], goal: List[float]):
    start = tuple(start)
    goal = tuple(goal)
    if not start or not goal:
            print("❌ ERROR: Missing start or goal coordinates")
            return JSONResponse(status_code=400,content={"error": "Both 'start' and 'goal' must be provided."})
    print("\n" + "="*60)
    print("🚀 NAVIGATION REQUEST RECEIVED")
    print("="*60)
    print(f"📍 Start: {start[0]:.6f}, {start[1]:.6f}")
    print(f"📍 Goal:  {goal[0]:.6f}, {goal[1]:.6f}")
    print("-"*60)
    # Step 1: Attempt with custom graph
    print("🔍 Attempting custom algorithm (A* with footpath data)...")
    nav = Navigator()
    path = nav.astar(NavigationGraph.graph, start, goal)

    if path:
        print(f"✅ SUCCESS: Custom algorithm found path!")
        print(f"📊 Path length: {len(path)} waypoints")
        print(f"🎯 Source: Custom Footpath Database")
        print("-"*60)
        print("🗺️  PATH COORDINATES:")
        for i, coord in enumerate(path, 1):
            print(f"   {i:3d}. [{coord[0]:.6f}, {coord[1]:.6f}]")
        print("="*60 + "\n")
        return JSONResponse(content={"source": "custom", "path": path},status_code=200)

    # Step 2: Fallback to Google Directions API
    print("⚠️  Custom algorithm failed - no path found in database")
    print("🔄 Falling back to Google Directions API...")
    google_path = get_google_directions(start, goal)
    if google_path:
        print(f"✅ SUCCESS: Google Directions API found path!")
        print(f"📊 Path length: {len(google_path)} waypoints")
        print(f"🎯 Source: Google Directions API")
        print("-"*60)
        print("🗺️  PATH COORDINATES:")
            # Print first 10, last 10, and total count if path is very long
        if len(google_path) <= 20:
                for i, coord in enumerate(google_path, 1):
                    print(f"   {i:3d}. [{coord[0]:.6f}, {coord[1]:.6f}]")
        else:
                for i, coord in enumerate(google_path[:10], 1):
                    print(f"   {i:3d}. [{coord[0]:.6f}, {coord[1]:.6f}]")
                print(f"   ... ({len(google_path) - 20} coordinates omitted) ...")
                for i, coord in enumerate(google_path[-10:], len(google_path) - 9):
                    print(f"   {i:3d}. [{coord[0]:.6f}, {coord[1]:.6f}]")
        print("="*60 + "\n")
        return JSONResponse(content={"source": "google", "path": google_path},status_code=200)

    print("❌ FAILURE: No path found from either source")
    print("="*60 + "\n")
    return JSONResponse(content={"error": "No path found"},status_code=404), 

def decode_polyline(polyline_str):
    coords = []
    index = lat = lng = 0

    while index < len(polyline_str):
        for coord in (lat, lng):
            result = shift = 0
            while True:
                b = ord(polyline_str[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            d = ~(result >> 1) if (result & 1) else (result >> 1)
            if coord is lat:
                lat += d
            else:
                lng += d
        coords.append((lat / 1e5, lng / 1e5))

    return coords

def get_google_directions(start, goal):
    url = (
        "https://maps.googleapis.com/maps/api/directions/json"
        f"?origin={start[0]},{start[1]}"
        f"&destination={goal[0]},{goal[1]}"
        f"&key={GOOGLE_API_KEY}"
    )

    res = requests.get(url)
    if res.status_code != 200:
        return None

    data = res.json()
    if not data.get("routes"):
        return None

    poly = data["routes"][0]["overview_polyline"]["points"]
    return decode_polyline(poly)



