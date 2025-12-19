from fastapi import APIRouter



api_router = APIRouter(
    tags=["Endpoints"]
)

@api_router.get("/health", summary="Health Check Endpoint")
async def health_check():
    return {"status": "ok"}
