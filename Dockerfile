FROM python:3.11-slim

WORKDIR /app

# Copiar los archivos de requisitos
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Crear directorio models si no existe
RUN mkdir -p models

# Crear un archivo de modelo falso para pruebas
RUN echo "Este es un archivo de modelo falso para pruebas." > models/fake_model.txt

# Exponer el puerto
EXPOSE 8001

# Comando para iniciar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
