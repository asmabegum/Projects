from pydantic import BaseModel



class ImageRequest(BaseModel):
    # text: str = None
    file_path: str = None
    num_images: int

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5