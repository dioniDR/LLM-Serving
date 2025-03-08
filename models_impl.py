"""
Implementación de modelos para la API LLM-Serving.
Este módulo contiene las clases para diferentes tipos de modelos.
"""

import os
import json
import numpy as np
from typing import Optional, Dict, Any, List

class OnnxModel:
    """Clase para modelos en formato ONNX."""
    
    def __init__(self, model_path: str):
        """
        Inicializa un modelo ONNX.
        
        Args:
            model_path: Ruta al archivo .onnx del modelo
        """
        # Importamos aquí para no requerir onnxruntime en toda la aplicación
        import onnxruntime as ort
        from transformers import AutoTokenizer
        
        print(f"Cargando modelo ONNX desde {model_path}...")
        self.session = ort.InferenceSession(model_path)
        
        # Obtener directorio del modelo para buscar archivos de configuración
        model_dir = os.path.dirname(model_path)
        
        # Intentar determinar el tokenizer adecuado
        tokenizer_name = None
        
        # Buscar en config.json
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            # Intentar obtener el nombre original del modelo
            if "_name_or_path" in config:
                tokenizer_name = config["_name_or_path"]
        
        if not tokenizer_name:
            # Si es un modelo Phi, usar el tokenizer de Phi
            if "phi" in os.path.basename(model_path).lower():
                tokenizer_name = "microsoft/phi-1_5"
            else:
                # Intentar usar el nombre del directorio como modelo en HF
                tokenizer_name = os.path.basename(model_dir)
        
        # Si hay archivos de tokenizer locales, usarlos
        if os.path.exists(os.path.join(model_dir, "tokenizer.json")):
            print(f"Usando tokenizer local desde {model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        else:
            # Cargar tokenizer desde Hugging Face
            print(f"Cargando tokenizer desde {tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Obtener nombres de inputs/outputs del modelo
        self.input_names = [input_meta.name for input_meta in self.session.get_inputs()]
        self.output_names = [output_meta.name for output_meta in self.session.get_outputs()]
        print(f"Input names: {self.input_names}")
        print(f"Output names: {self.output_names}")
    
    def predict(self, prompt: str, max_length: int = 200, temperature: float = 0.7) -> str:
        """
        Genera texto a partir de un prompt.
        
        Args:
            prompt: Texto de entrada
            max_length: Longitud máxima de la respuesta
            temperature: Temperatura para la generación (mayor valor = más aleatorio)
            
        Returns:
            Texto generado
        """
        # Tokenizar la entrada
        inputs = self.tokenizer(prompt, return_tensors="np")
        
        # Preparar los inputs para ONNX Runtime
        onnx_inputs = {}
        for name in self.input_names:
            if name in inputs:
                onnx_inputs[name] = inputs[name]
            elif name == "attention_mask":
                onnx_inputs[name] = np.ones_like(inputs["input_ids"])
        
        # Iniciar generación
        input_ids = inputs["input_ids"]
        current_length = input_ids.shape[1]
        
        # Generación token por token
        for _ in range(max_length - current_length):
            # Hacer inferencia con el modelo
            outputs = self.session.run(self.output_names, onnx_inputs)
            
            # Obtener el siguiente token (último token de la secuencia)
            next_token_logits = outputs[0][:, -1, :]
            
            # Aplicar temperatura si es necesario
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Seleccionar el token con mayor probabilidad
            next_token = np.argmax(next_token_logits, axis=-1)
            
            # Añadir el token generado a la secuencia de entrada
            input_ids = np.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)
            
            # Actualizar inputs para la siguiente iteración
            onnx_inputs["input_ids"] = input_ids
            if "attention_mask" in onnx_inputs:
                onnx_inputs["attention_mask"] = np.ones_like(input_ids)
            
            # Parar si generamos un token de fin
            if next_token[0] == self.tokenizer.eos_token_id:
                break
        
        # Decodificar la secuencia generada
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

# Clase específica para Phi-1.5 ONNX
class Phi15OnnxModel(OnnxModel):
    """Clase específica para el modelo Phi-1.5 ONNX."""
    
    def __init__(self):
        """Inicializa el modelo Phi-1.5 ONNX desde la ubicación predeterminada."""
        super().__init__("models/phi-1.5-onnx/model.onnx")

# Diccionario para mapear modelos a sus implementaciones
MODEL_IMPLEMENTATIONS = {
    "phi-1.5-onnx": Phi15OnnxModel
}

def get_model_implementation(model_name: str):
    """
    Obtiene la clase de implementación para un modelo específico.
    
    Args:
        model_name: Nombre del modelo
        
    Returns:
        Clase de implementación del modelo o None si no existe
    """
    return MODEL_IMPLEMENTATIONS.get(model_name)
