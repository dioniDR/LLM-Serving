#!/bin/bash

# Script para configurar el proyecto LLM-Serving

# Crear directorios necesarios
mkdir -p models

# Crear un archivo de modelo falso para pruebas
echo "Este es un archivo de modelo falso para pruebas." > models/fake_model.txt

# Verificar si Python está instalado
if command -v python3 &>/dev/null; then
    echo "✅ Python encontrado"
    
    # Crear entorno virtual
    echo "🔧 Creando entorno virtual..."
    python3 -m venv venv
    
    # Activar entorno virtual
    source venv/bin/activate
    
    # Instalar dependencias
    echo "📦 Instalando dependencias..."
    pip install -r requirements.txt
    
    echo "🚀 Configuración completada. Para iniciar el servidor:"
    echo "source venv/bin/activate"
    echo "uvicorn main:app --reload"
else
    echo "❌ Python no encontrado. Por favor, instala Python 3.8+ e intenta nuevamente."
fi
