import sys
import os
import types

# Completely remove flash_attn if it exists
if "flash_attn" in sys.modules:
    del sys.modules["flash_attn"]

# Create a proper mock module with __spec__
class FlashAttnMock(types.ModuleType):
    """Mock flash_attn module to prevent import errors"""
    def __init__(self):
        super().__init__("flash_attn")
        self.__version__ = "0.0.0"
        # Create a proper __spec__ that won't trigger the error
        self.__spec__ = types.SimpleNamespace(
            name="flash_attn",
            loader=None,
            origin="mock",
            submodule_search_locations=None,
            cached=None,
            parent="flash_attn"
        )

# Install the mock before ANY transformers imports
sys.modules["flash_attn"] = FlashAttnMock()
os.environ["DISABLE_FLASH_ATTENTION"] = "1"

# Now import transformers and patch the check function BEFORE it's used
import transformers.utils.import_utils

# Override the function that checks for flash_attn
def patched_is_package_available(pkg_name):
    """Patched version that returns False for flash_attn"""
    if pkg_name == "flash_attn":
        return False
    # For other packages, do the normal check
    import importlib.util
    try:
        return importlib.util.find_spec(pkg_name) is not None
    except (ImportError, ValueError, AttributeError):
        return False

# Apply the patch
transformers.utils.import_utils._is_package_available = patched_is_package_available

# Now safe to import everything else
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, status
from pydantic import BaseModel
from typing import Optional
import base64
import io
from PIL import Image
import torch
import numpy as np

from utils import (
    check_ocr_box,
    get_yolo_model,
    get_caption_model_processor,
    get_som_labeled_img,
)
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForCausalLM

# Initialize FastAPI
app = FastAPI()

# API Key from environment variable
API_KEY = os.getenv("API_KEY", "CHANGE-ME-IMMEDIATELY")

# Global variables for models
yolo_model = None
caption_model_processor = None


class ProcessResponse(BaseModel):
    image: str
    parsed_content_list: str
    label_coordinates: str


def verify_api_key(x_api_key: str = Header(...)):
    """Verify the API key from request header"""
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return x_api_key


@app.on_event("startup")
async def load_models():
    """Load models on application startup"""
    global yolo_model, caption_model_processor
    
    print("Loading YOLO model...")
    try:
        yolo_model = YOLO("weights/icon_detect/best.pt").to("cuda")
        print("✓ YOLO model loaded on CUDA")
    except Exception as e:
        print(f"⚠ CUDA not available, loading YOLO on CPU: {e}")
        yolo_model = YOLO("weights/icon_detect/best.pt")
    
    print("Loading Florence-2 processor...")
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-base", 
        trust_remote_code=True
    )
    
    print("Loading Florence-2 model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "weights/icon_caption_florence",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="sdpa"
        ).to("cuda")
        print("✓ Florence-2 model loaded on CUDA")
    except Exception as e:
        print(f"⚠ CUDA not available, loading Florence-2 on CPU: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            "weights/icon_caption_florence",
            torch_dtype=torch.float32,
            trust_remote_code=True,
            attn_implementation="eager"
        )
    
    caption_model_processor = {"processor": processor, "model": model}
    print("✓ All models loaded successfully!")


def process(
    image_input: Image.Image, box_threshold: float, iou_threshold: float
) -> ProcessResponse:
    """Process image with YOLO and Florence-2 models"""
    if yolo_model is None or caption_model_processor is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    # Create imgs directory if needed
    os.makedirs("imgs", exist_ok=True)
    
    image_save_path = "imgs/saved_image_demo.png"
    image_input.save(image_save_path)
    image = Image.open(image_save_path)
    
    box_overlay_ratio = image.size[0] / 3200
    draw_bbox_config = {
        "text_scale": 0.8 * box_overlay_ratio,
        "text_thickness": max(int(2 * box_overlay_ratio), 1),
        "text_padding": max(int(3 * box_overlay_ratio), 1),
        "thickness": max(int(3 * box_overlay_ratio), 1),
    }

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
        image_save_path,
        display_img=False,
        output_bb_format="xyxy",
        goal_filtering=None,
        easyocr_args={"paragraph": False, "text_threshold": 0.9},
        use_paddleocr=True,
    )
    text, ocr_bbox = ocr_bbox_rslt
    
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_save_path,
        yolo_model,
        BOX_TRESHOLD=box_threshold,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=caption_model_processor,
        ocr_text=text,
        iou_threshold=iou_threshold,
    )
    
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    parsed_content_list_str = "\n".join(parsed_content_list)

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return ProcessResponse(
        image=img_str,
        parsed_content_list=str(parsed_content_list_str),
        label_coordinates=str(label_coordinates),
    )


@app.get("/health")
async def health_check():
    """Health check endpoint - no auth required"""
    models_loaded = yolo_model is not None and caption_model_processor is not None
    return {
        "status": "healthy" if models_loaded else "loading",
        "models_loaded": models_loaded
    }


@app.post("/process_image", response_model=ProcessResponse)
async def process_image(
    image_file: UploadFile = File(...),
    box_threshold: float = 0.05,
    iou_threshold: float = 0.1,
    api_key: str = Header(..., alias="X-API-Key")  # Require API key
):
    """Process uploaded image for icon detection and captioning"""
    # Verify API key
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    try:
        contents = await image_file.read()
        image_input = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    try:
        response = process(image_input, box_threshold, iou_threshold)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")