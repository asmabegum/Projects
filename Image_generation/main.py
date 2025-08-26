import logging
import uvicorn
from fastapi import FastAPI, Request
from dotenv import load_dotenv
import os
import warnings
import logging
from routes.routesInference import router as rag_router
warnings.filterwarnings("ignore", category=DeprecationWarning)

#Loading environment variables
load_dotenv(".env")
env = os.environ.get("APIENV")
load_dotenv(f".app/env/.en{env}app")

#Setting up logging
log_level = logging.INFO if os.environ.get("LOGLEVEL") == "DEBUG" else logging.ERROR

# configure FastAPI 
app = FastAPI(
    title="AI Image Generation API",
    description="This is my AI Image Generation API",
    version="1.0.0"
)

@app.on_event("startup")
def startup_app():
    """
    This function is called when the application starts up.
    It can be used to perform any initialization tasks.
    """
    logging.basicConfig(level=log_level)
    logging.info("Starting up the application...")

@app.on_event("shutdown")
def shutdown_app():
    """
    This function is called when the application shuts down.
    It can be used to perform any cleanup tasks.
    """
    logging.info("Shutting down the application...")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log requests and responses.
    """
    logging.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logging.info(f"Response: {response.status_code}")
    return response

app.include_router(rag_router)

if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port = 8000)