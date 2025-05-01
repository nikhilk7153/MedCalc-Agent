from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
import os
from dotenv import load_dotenv
import uvicorn

# Import routes
from routes import router

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MedCalc-Agent API",
    description="REST API for MedCalc-Agent - A browser-augmented LLM agent for medical calculations",
    version="1.0.0"
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates for HTML rendering
templates = Jinja2Templates(directory="templates")

# Include all routes
app.include_router(router)

# Home route
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    """Render the home page"""
    return templates.TemplateResponse(
        "index.html", 
        {"request": request}
    )

# Run the application
if __name__ == "__main__":
    # Limit reload to only routes directory for more stability
    # Exclude browser_calculator.py from reload to prevent interruptions
    reload_dirs = ["routes", "templates"]
    
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        reload_dirs=reload_dirs
    ) 