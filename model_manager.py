import json
import os
from typing import Dict, List, Optional, Any
import importlib.util

# Ruta al archivo de configuración de modelos
CONFIG_FILE = "model_config.json"

# Diccionario para almacenar instancias de modelos cargados
_loaded_models = {}

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
    
    # Verificar si es un modelo implementado directamente (como phi-2)
    if is_implemented_model(model_name):
        return "IMPLEMENTED"
    
    return None

def list_available_models() -> List[str]:
    """Lista todos los modelos disponibles en la configuración."""
    config = load_model_config()
    
    # Obtener modelos implementados programáticamente
    try:
        # Importar dinámicamente el módulo models_impl
        if importlib.util.find_spec("models_impl") is not None:
            models_impl = importlib.import_module("models_impl")
            implemented_models = getattr(models_impl, "MODEL_IMPLEMENTATIONS", {}).keys()
        else:
            implemented_models = []
    except ImportError:
        implemented_models = []
    
    # Combinar modelos de configuración y modelos implementados
    models = [
        model_name for model_name, path in config.items()
        if model_name == "fake_model" or os.path.exists(path)
    ]
    models.extend(implemented_models)
    
    return list(set(models))  # Eliminar duplicados

def is_implemented_model(model_name: str) -> bool:
    """Verifica si el modelo tiene una implementación programática."""
    try:
        if importlib.util.find_spec("models_impl") is not None:
            models_impl = importlib.import_module("models_impl")
            return model_name in getattr(models_impl, "MODEL_IMPLEMENTATIONS", {})
        return False
    except ImportError:
        return False

def load_model(model_name: str) -> Optional[Any]:
    """
    Carga un modelo en memoria si no está ya cargado.
    
    Args:
        model_name (str): Nombre del modelo a cargar
        
    Returns:
        Any: Instancia del modelo cargado o None si no se puede cargar
    """
    # Si el modelo ya está cargado, devolverlo
    if model_name in _loaded_models:
        return _loaded_models[model_name]
    
    # Verificar si es un modelo implementado directamente
    if is_implemented_model(model_name):
        try:
            models_impl = importlib.import_module("models_impl")
            model_class = getattr(models_impl, "MODEL_IMPLEMENTATIONS")[model_name]
            _loaded_models[model_name] = model_class()
            return _loaded_models[model_name]
        except (ImportError, KeyError, Exception) as e:
            print(f"Error al cargar el modelo {model_name}: {str(e)}")
            return None
    
    # Para otros tipos de modelos en el futuro
    # (GGUF, ONNX, etc.)
    
    return None

def inference(model_name: str, prompt: str) -> Optional[str]:
    """
    Realiza inferencia con un modelo.
    
    Args:
        model_name (str): Nombre del modelo
        prompt (str): Texto de entrada para el modelo
        
    Returns:
        str: Texto generado por el modelo o None si hay error
    """
    # Modelo de simulación (fake_model)
    if model_name == "fake_model":
        return f"Simulación: '{prompt}' procesado por {model_name}"
    
    # Cargar el modelo si no está cargado
    model = load_model(model_name)
    if model is None:
        return None
    
    # Realizar predicción según el tipo de modelo
    try:
        # Para modelos implementados directamente
        if is_implemented_model(model_name):
            return model.predict(prompt)
        
        # Para otros tipos de modelos en el futuro
        
    except Exception as e:
        print(f"Error en inferencia con {model_name}: {str(e)}")
        return None

def unload_model(model_name: str) -> bool:
    """
    Libera recursos de un modelo cargado.
    
    Args:
        model_name (str): Nombre del modelo a descargar
        
    Returns:
        bool: True si se descargó correctamente, False en caso contrario
    """
    if model_name in _loaded_models:
        try:
            del _loaded_models[model_name]
            return True
        except Exception:
            return False
    return False
