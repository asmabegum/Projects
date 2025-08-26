from PIL import Image
import faiss
import numpy as np
import uuid
import requests
from io import BytesIO
import pandas as pd
from tqdm import tqdm
import os
#from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('clip-ViT-B-32')
def download_image_from_url(image_url: str):
    """
    Download an image from a URL and return it as a PIL Image object.
    
    :param image_url: URL of the image.
    :return: PIL Image object.
    """
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an error for bad responses
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        raise RuntimeError(f"Error downloading image: {str(e)}")

def download_images_and_caption(file_path, num_images=None, target_size=(800, 800)):
    """
    Download and optimize images from photos.csv
    
    Args:
        num_images: Number of images to download (default: all images in CSV)
        output_dir: Directory to save images (default: 'images')
        target_size: Max image dimensions (default: (800, 800))
    """
    doc_id = str(uuid.uuid4())
    #vectorstore, collection = initialise_vector_store()
    output_dir = "./downloaded_images"
    output_index_dir = "./faiss_index/faiss"
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV and prepare dataset
    df = pd.read_csv(file_path, usecols=['photo_image_url'])
    id_key = "doc_id"
    # Ensure the DataFrame is not empty
    if num_images:
        df = df.head(num_images)
    
    # Download images
    #print(f"Downloading {len(df)} images to {output_dir}...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # # Download and save image
            filename = f"{(idx+1):04d}.jpg"
            output_image_path = os.path.join(output_dir, filename)
            image_path = row['photo_image_url']
            response = requests.get(row['photo_image_url'], timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                img.thumbnail(target_size, Image.Resampling.LANCZOS)
                # plt.imshow(img)
                # plt.show()
                img.save(output_image_path, 'JPEG', quality=85, optimize=True)
                print(f"Downloaded and saved image {idx+1}: {output_image_path}")
                # Embed and store the image with doc_id
                embeddings = embed_image_with_clip(img, model)
                # store_embedding_with_id(image_vector, doc_id)
                add_documents_index(embeddings= embeddings, image_paths = output_image_path,output_path= output_index_dir)
                
        except Exception as e:
            print(f"Error")


def embed_image_with_clip(image, model):
    """
    Embed an image using the CLIP model.
    
    :param image_path: Path to the image file.
    :return: A vector representation of the image.
    """
    try:
        embeddings = []
        #initialise_vector_store()
        #image_vector = np.array(image)
        embedding = model.encode(image)
        embeddings.append(embedding)
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Error embedding image: {str(e)}")
        
def add_documents_index(embeddings, image_paths, output_path):
    """
    Adds document embeddings and their corresponding image paths to a FAISS index.
    This function creates or loads a FAISS index, adds the provided embeddings with unique IDs,
    and saves the updated index to the specified output path. It also appends the image paths
    to a separate file for reference.
    Args:
        embeddings (list or np.ndarray): A list or NumPy array of embeddings to be added to the index.
        image_paths (str): A string containing the paths of the images corresponding to the embeddings.
        output_path (str): The file path where the FAISS index will be saved.
    Returns:
        faiss.Index: The updated FAISS index.
    Raises:
        RuntimeError: If there is an error during the process of adding documents to the FAISS index.
    Notes:
        - The embeddings are expected to be in a NumPy array format with dtype `float32`.
        - The FAISS index uses inner product similarity for vector comparisons.
        - The image paths are saved in a separate file with the `.paths` extension.
    """

    try:
        # Ensure embeddings are in NumPy array format
        vectors = np.array(embeddings).astype(np.float32)

        # Check if the FAISS index already exists
        if os.path.exists(output_path):
            # Load the existing index
            index = faiss.read_index(output_path)
            print(f"Loaded existing FAISS index from {output_path}")
        else:
            # Create a new FAISS index
            dimension = vectors.shape[1]  # Dimension of the embeddings
            index = faiss.IndexFlatIP(dimension)  # Inner product similarity
            index = faiss.IndexIDMap(index)
            print(f"Created a new FAISS index at {output_path}")
    # Debug: Print embeddings
        print(f"Embeddings: {embeddings}")
        print(f"Number of embeddings: {len(embeddings)}")
        print(f"Image paths: {image_paths}")
        
        start_id = index.ntotal
        ids = np.array(range(start_id, start_id + len(embeddings)), dtype=np.int64)

        # Add vectors to the index with IDs
        index.add_with_ids(vectors, ids)
        
        # Save the index
        faiss.write_index(index, output_path)
        print(f"Index created and saved to {output_path}")
        # Save image paths
        with open(output_path + '.paths', 'a') as f:
            f.write(image_paths + '\n')
        
        return index
    except Exception as e:
            raise RuntimeError(f"Error adding documents to FAISS index: {str(e)}")

      