import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

output_index_dir = "./faiss_index/faiss"
model = SentenceTransformer('clip-ViT-B-32')
# Load BLIP model and processor
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def load_faiss_index(index_path):
    index = faiss.read_index(index_path)
    with open(index_path + '.paths', 'r') as f:
        image_paths = [line.strip() for line in f]
    print(f"Index loaded from {index_path}")
    print(f"Number of embeddings in the index: {index.ntotal}")
    print(f"Number of image paths: {len(image_paths)}")
    return index, image_paths

def generate_image_explanation_with_blip(image_path):
    """
    Generate a detailed explanation for an image using the LLaVA model.
    
    :param image_path: Path to the image file.
    :return: A string containing the explanation for the image.
    """
    try:
        # Load the image
        image = Image.open(image_path).convert("RGB")
        
        # Process the image and generate a caption
        inputs = blip_processor(images=image, return_tensors="pt")
        outputs = blip_model.generate(**inputs)
        explanation = blip_processor.decode(outputs[0], skip_special_tokens=True)
        
        return explanation
    except Exception as e:
        raise RuntimeError(f"Error generating explanation for image {image_path}: {str(e)}")

def retrieve_similar_images(query, top_k=5):
    """
    Retrieve the top-k most similar images to a given query and generate explanations for them.
    Args:
        query (str): The input query string to find similar images for.
        top_k (int, optional): The number of top similar images to retrieve. Defaults to 5.
    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains:
            - "image_path" (str): The file path of the retrieved image.
            - "explanation" (str): The explanation generated for the image.
    Notes:
        - This function uses a FAISS index to search for similar images based on the query.
        - The `model.encode` method is used to generate feature embeddings for the query.
        - The `generate_image_explanation_with_blip` function is used to generate explanations for each retrieved image.
        - If an index value of -1 is encountered, it is skipped as it indicates an invalid or missing result.
    """
    
    index, image_paths = load_faiss_index(output_index_dir)    
    query_features = model.encode(query)
    query_features = query_features.astype(np.float32).reshape(1, -1)

    distances, indices = index.search(query_features, top_k)
    print(f"Indices of top {top_k} similar images: {indices}")

    # Retrieve image paths and generate explanations
    retrieved_images = []
    for idx in indices[0]:
        if idx == -1:
            continue
        img_path = image_paths[int(idx)]
        explanation = generate_image_explanation_with_blip(img_path)
        retrieved_images.append({"image_path": img_path, "explanation": explanation})

    return retrieved_images



    