# Backend Dockerfile
FROM python:3.10-slim

# Instalar dependencias del sistema (incluyendo OpenCV y audio)
RUN apt-get update && apt-get install -y \
    libspatialindex-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements.txt (desde el contexto raíz)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente necesario
COPY src/ ./src/
COPY api/ ./api/

# Exponer puerto 8000 para FastAPI
EXPOSE 8000

# Comando para ejecutar  la API
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
