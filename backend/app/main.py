from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.endpoints import api_router
from app.utils.load_model import load_best_model_from_mlflow


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        import app.api.endpoints as endpoints_module
        model, info = load_best_model_from_mlflow(
            experiment_name="sales_forecasting",
            model_name="BestRegressionModel"
        )
        endpoints_module.loaded_model = model
        endpoints_module.model_info = info
        print(f"✅ Model loaded: {info}")
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    print("Shutting down...")


app = FastAPI(title="Sales Forecasting API", lifespan=lifespan)
app.include_router(api_router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)