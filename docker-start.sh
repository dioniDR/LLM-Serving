#!/bin/bash

# Script para construir y ejecutar el proyecto usando Docker

# Asegurarse de que el directorio de modelos existe
mkdir -p models

# Crear un archivo de modelo falso para pruebas si no existe
if [ ! -f models/fake_model.txt ]; then
  echo "Este es un archivo de modelo falso para pruebas." > models/fake_model.txt
  echo "✅ Creado modelo falso para pruebas"
fi

# Verificar si Docker está instalado
if command -v docker &>/dev/null; then
  echo "✅ Docker encontrado"
  
  # Verificar si docker-compose está instalado
  if command -v docker-compose &>/dev/null; then
    echo "✅ Docker Compose encontrado"
    
    # Construir y levantar los contenedores
    echo "🔧 Construyendo y ejecutando contenedores..."
    docker-compose up --build -d
    
    echo "🚀 Servicios iniciados. La API está disponible en: http://localhost:8001"
    echo "📝 Para ver logs: docker-compose logs -f"
    echo "⏹️ Para detener: docker-compose down"
  else
    echo "❌ Docker Compose no encontrado."
    echo "🔧 Construyendo y ejecutando solo con Docker..."
    
    # Construir la imagen
    docker build -t llm-serving .
    
    # Ejecutar el contenedor
    docker run -d --name llm-serving -p 8001:8001 -v "$(pwd)/models:/app/models" llm-serving
    
    echo "🚀 Servicios iniciados. La API está disponible en: http://localhost:8000"
    echo "📝 Para ver logs: docker logs -f llm-serving"
    echo "⏹️ Para detener: docker stop llm-serving && docker rm llm-serving"
  fi
else
  echo "❌ Docker no encontrado. Por favor, instala Docker e intenta nuevamente."
fi
