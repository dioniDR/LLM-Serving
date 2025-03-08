from fastapi import APIRouter, HTTPException
from model_manager import get_model_path, list_available_models

router = APIRouter()

@router.get("/models")
def get_available_models():
    """Lista todos los modelos disponibles en la configuración."""
    models = list_available_models()
    return {"models": models}

@router.get("/query/{model_name}")
def query_model(model_name: str, prompt: str):
    """Consulta el modelo especificado (simulado si no hay modelo real)."""
    model_path = get_model_path(model_name)
    
    if not model_path:
        raise HTTPException(status_code=404, detail=f"Modelo '{model_name}' no encontrado")

    # Simulación de respuesta sin modelo real
    response = f"Simulación: '{prompt}' procesado por {model_name}"
    
    # En el futuro, aquí se implementará la llamada real al modelo:
    # response = load_and_inference(model_path, prompt)
    
    return {"model": model_name, "response": response}
