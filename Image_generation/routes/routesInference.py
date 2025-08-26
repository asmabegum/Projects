from fastapi import FastAPI, HTTPException, APIRouter
import base64
import os
from models.modelsInference import ImageRequest, QueryRequest
from lib.image_storing import download_images_and_caption
from lib.generate_llm_response import retrieve_similar_images
import base64

def encode_image_to_base64(image_path):
    """
    Encode an image to a Base64 string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


router = APIRouter()

@router.post("/store_image", tags=["store_image"])
async def store_image(image_request: ImageRequest):
    try:
        download_images_and_caption(image_request.file_path, image_request.num_images)
        return {"message": "Image ingested and caption generated successfully."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/generate_image", tags=["image_search"])
async def generate_image(query_request: QueryRequest):
    try:
        retrieved_images = retrieve_similar_images(query_request.query, top_k=query_request.top_k)
        # Encode images as Base64
        for img_data in retrieved_images:
            # to compress image size, convert to webp and then to base64
            img_data["base64_image"] = encode_image_to_base64(img_data["image_path"])
        return {"query": query_request.query, 
                "retrieved_images": retrieved_images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

