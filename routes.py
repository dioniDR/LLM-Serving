from fastapi import APIRouter, HTTPException
from model_manager import get_model_path, list_available_models, inference

router = APIRouter()

@router.get("/models")
def get_available_models():
    """Lista todos los modelos disponibles en la configuración."""
    models = list_available_models()
    return {"models": models}

@router.get("/query/{model_name}")
def query_model(model_name: str, prompt: str):
    """Consulta el modelo especificado."""
    model_path = get_model_path(model_name)
    
    if not model_path:
        raise HTTPException(status_code=404, detail=f"Modelo '{model_name}' no encontrado")

    # Realizar inferencia con el modelo
    response = inference(model_name, prompt)
    
    if response is None:
        raise HTTPException(status_code=500, detail=f"Error al procesar la consulta con el modelo '{model_name}'")
    
    return {"model": model_name, "response": response}
