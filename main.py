from fastapi import FastAPI
from routes import router

app = FastAPI(title="LLM-Serving", 
              description="API ligera para servir modelos de lenguaje",
              version="0.1.0")

# Incluir las rutas definidas en routes.py
app.include_router(router)

# Ruta raíz para verificar que la API está funcionando
@app.get("/")
def read_root():
    return {"message": "Bienvenido a LLM-Serving - API para modelos de lenguaje"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
