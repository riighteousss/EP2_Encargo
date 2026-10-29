FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copia también modelo_cnn.keras y clases.json (salidas del notebook) a esta carpeta antes de construir
COPY main.py modelo_cnn.keras clases.json ./
ENV PORT=8000
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
