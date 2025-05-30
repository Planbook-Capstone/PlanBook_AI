from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.endpoints import ai_endpoints, chemistry_endpoints

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
    debug=settings.DEBUG
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create API router
api_router = APIRouter()

# Include routers from endpoints
api_router.include_router(ai_endpoints.router, prefix="/ai", tags=["AI Services"])
api_router.include_router(chemistry_endpoints.router, prefix="/chemistry", tags=["Chemistry Services"])

# Add API router to app
app.include_router(api_router, prefix=settings.API_PREFIX)

@app.get("/")
async def root():
    return {"message": "Welcome to PlanBook AI Service", "docs": f"{settings.API_PREFIX}/docs"}
