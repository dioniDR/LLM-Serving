# LLM-Serving

Una API ligera y eficiente para servir modelos de lenguaje de gran escala (LLMs) localmente.

## 🚀 Descripción

LLM-Serving es una API REST minimalista diseñada para cargar y consultar modelos de lenguaje de manera sencilla en un entorno local. Este proyecto permite exponer diversos modelos en formato GGUF, ONNX y otros formatos comunes a través de una interfaz HTTP unificada.

## ✨ Características

- **Arquitectura ligera**: Diseñada para ser simple pero efectiva.
- **Multi-modelo**: Soporte para servir múltiples modelos simultáneamente.
- **Flexibilidad de formatos**: Compatible con modelos en formato GGUF, ONNX y más.
- **Modo simulación**: Capacidad de probar la API sin modelos reales.
- **Fácil configuración**: Gestión de modelos a través de un archivo JSON.

## 📋 Requisitos

- Python 3.8+
- FastAPI
- Uvicorn
- Dependencias específicas para los modelos (se detallan en `requirements.txt`)

## 🔧 Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/llm-serving.git
cd llm-serving
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Configurar modelos en `model_config.json`:
```json
{
  "mistral-7b": "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
  "phi-2": "models/phi-2.onnx",
  "fake_model": "models/fake_model.txt"
}
```

## 🚀 Uso

### Iniciar el servidor

```bash
uvicorn main:app --reload
```

### Listar modelos disponibles

```bash
curl http://localhost:8000/models
```

### Consultar un modelo

```bash
curl "http://localhost:8000/query/mistral-7b?prompt=Explica la teoría de la relatividad"
```

### Modo de simulación

Para probar la API sin modelos reales:

1. Crear un archivo de prueba:
```bash
touch models/fake_model.txt
```

2. Agregar al archivo de configuración:
```json
{
  "fake_model": "models/fake_model.txt"
}
```

3. Consultar el modelo simulado:
```bash
curl "http://localhost:8000/query/fake_model?prompt=Hola"
```

## 🛠️ Estructura del proyecto

```
llm-serving/
├── main.py                # Punto de entrada de la aplicación
├── routes.py              # Definición de rutas API
├── model_manager.py       # Gestión de modelos
├── model_config.json      # Configuración de modelos
├── models/                # Directorio para almacenar modelos
│   ├── fake_model.txt     # Modelo de prueba
│   └── ...                # Otros modelos
└── requirements.txt       # Dependencias
```

## 📝 Planes futuros

- Implementar streaming de respuestas
- Añadir soporte para parámetros de generación avanzados
- Desarrollar una interfaz web simple
- Optimizar la inferencia para modelos grandes

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para más detalles.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir los cambios importantes antes de enviar un pull request.
