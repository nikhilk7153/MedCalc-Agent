# Initialize the routes package
from fastapi import APIRouter
from routes.api import router as api_router
from routes.websocket import router as websocket_router

# Create a combined router
router = APIRouter()

# Include the API routes
router.include_router(api_router)

# Include the WebSocket routes
router.include_router(websocket_router) 