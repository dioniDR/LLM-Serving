import json
import os
from typing import Dict, List, Optional

# Ruta al archivo de configuración de modelos
CONFIG_FILE = "model_config.json"

def load_model_config() -> Dict[str, str]:
    """Carga la configuración de modelos desde el archivo JSON."""
    try:
        with open(CONFIG_FILE, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        # Si el archivo no existe o está mal formateado, devolver un diccionario vacío
        return {}

def get_model_path(model_name: str) -> Optional[str]:
    """Obtiene la ruta al archivo del modelo especificado."""
    config = load_model_config()
    model_path = config.get(model_name)
    
    # Verificar si el modelo existe en la configuración y si el archivo existe
    if model_path and os.path.exists(model_path):
        return model_path
    
    # Verificar si existe el archivo fake_model.txt para simulación
    if model_name == "fake_model" and "fake_model" in config:
        # Para el modelo fake, no es necesario que exista realmente el archivo
        return config["fake_model"]
    
    return None

def list_available_models() -> List[str]:
    """Lista todos los modelos disponibles en la configuración."""
    config = load_model_config()
    # Filtrar modelos cuyos archivos existen (excepto fake_model)
    return [
        model_name for model_name, path in config.items()
        if model_name == "fake_model" or os.path.exists(path)
    ]

# Funciones para agregar en el futuro:
# - load_model: Cargar un modelo en memoria
# - inference: Realizar inferencia con un modelo cargado
# - unload_model: Liberar recursos de un modelo cargado
