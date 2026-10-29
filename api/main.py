"""API de reconocimiento de personas — EP2 AVY1103 Deep Learning.

Recibe la matriz 128x128x3 (RGB, uint8) que genera la app móvil (iOS / Android),
ejecuta la CNN entrenada y responde el nombre de la persona o que no está en la base de datos.
"""
import base64
import binascii
import csv
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

TAMANO_IMAGEN = 128
CANALES = 3
RUTA_MODELO = os.getenv("RUTA_MODELO", "modelo_cnn.keras")
RUTA_METADATOS = os.getenv("RUTA_METADATOS", "clases.json")
RUTA_SUJETOS = os.getenv("RUTA_SUJETOS", "sujetos.csv")

logging.basicConfig(level=logging.INFO)
registro = logging.getLogger("api_reconocimiento")

MODELO = None                    # modelo Keras cargado al iniciar
NOMBRES_CLASES: List[str] = []   # nombre de cada clase, en el orden de salida del modelo
MAPA_SUJETOS: Dict[str, str] = {} # mapeo id_carpeta -> nombre_original (ej: persona001 -> Juan Pérez)
UMBRAL_CONFIANZA = 0.6           # bajo este valor se responde "no está en la base de datos"


def cargar_mapa_sujetos() -> Dict[str, str]:
    """Carga sujetos.csv si existe (del dataset colaborativo de visión)."""
    mapa = {}
    if os.path.exists(RUTA_SUJETOS):
        try:
            with open(RUTA_SUJETOS, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for fila in reader:
                    id_carp = fila.get("id_carpeta", "").strip()
                    nombre = fila.get("nombre_original") or fila.get("nombre_normalizado", "").strip()
                    if id_carp and nombre:
                        mapa[id_carp] = nombre
            registro.info("Mapeo de sujetos cargado desde %s: %d personas", RUTA_SUJETOS, len(mapa))
        except Exception as e:
            registro.warning("No se pudo parsear %s: %s", RUTA_SUJETOS, e)
    return mapa


def cargar_modelo() -> None:
    """Carga el modelo y sus metadatos (nombres de clases y umbral de confianza)."""
    global MODELO, NOMBRES_CLASES, UMBRAL_CONFIANZA, MAPA_SUJETOS
    import tensorflow as tf  # import perezoso: acelera el arranque y facilita las pruebas
    
    MAPA_SUJETOS = cargar_mapa_sujetos()
    
    MODELO = tf.keras.models.load_model(RUTA_MODELO)
    with open(RUTA_METADATOS, encoding="utf-8") as archivo:
        metadatos = json.load(archivo)
    
    clases_crudas = metadatos["clases"]
    # Si las clases tienen guiones bajos o son ids de sujetos.csv, formatear a nombre legible
    NOMBRES_CLASES = [
        MAPA_SUJETOS.get(c, c).replace("_", " ") for c in clases_crudas
    ]
    UMBRAL_CONFIANZA = float(metadatos.get("umbral", 0.6))
    registro.info("Modelo cargado: %d clases, umbral %.2f", len(NOMBRES_CLASES), UMBRAL_CONFIANZA)


@asynccontextmanager
async def ciclo_de_vida(_app: FastAPI):
    try:
        cargar_modelo()
    except Exception:
        registro.exception("No se pudo cargar el modelo al iniciar")  # /health lo reportará
    yield


app = FastAPI(
    title="API Reconocimiento DUOC UC",
    description="API de Visión Artificial para clasificación de rostros (Compatible con Android e iOS)",
    version="1.1.0",
    lifespan=ciclo_de_vida,
)

# Habilitar CORS para permitir llamadas desde cualquier cliente (iOS, Android, Web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EntradaImagen(BaseModel):
    imagen_base64: Optional[str] = None            # bytes crudos de la matriz 128x128x3, en base64
    matriz: Optional[List[List[List[int]]]] = None  # alternativa: lista anidada [128][128][3]


class RespuestaPrediccion(BaseModel):
    reconocido: bool
    nombre: Optional[str]
    confianza: float
    mensaje: str


def extraer_matriz(entrada: EntradaImagen) -> np.ndarray:
    """Valida la entrada y la entrega como arreglo uint8 de forma (128, 128, 3)."""
    forma_esperada = (TAMANO_IMAGEN, TAMANO_IMAGEN, CANALES)
    if entrada.imagen_base64 is not None:
        try:
            bytes_imagen = base64.b64decode(entrada.imagen_base64, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(status_code=400, detail="imagen_base64 no es un base64 válido")
        if len(bytes_imagen) != int(np.prod(forma_esperada)):
            raise HTTPException(status_code=400, detail=f"Se esperaban {int(np.prod(forma_esperada))} bytes (128x128x3)")
        return np.frombuffer(bytes_imagen, dtype=np.uint8).reshape(forma_esperada)
    if entrada.matriz is not None:
        try:
            matriz = np.array(entrada.matriz, dtype=np.int64)
        except ValueError:
            raise HTTPException(status_code=400, detail="La matriz no tiene forma regular")
        if matriz.shape != forma_esperada:
            raise HTTPException(status_code=400, detail=f"La matriz debe ser {forma_esperada}, llegó {matriz.shape}")
        if matriz.min() < 0 or matriz.max() > 255:
            raise HTTPException(status_code=400, detail="Los valores deben estar entre 0 y 255")
        return matriz.astype(np.uint8)
    raise HTTPException(status_code=400, detail="Debe enviar 'imagen_base64' o 'matriz'")


@app.get("/")
def inicio():
    """Ruta raíz para verificar que la API está en línea."""
    return {
        "servicio": "API Reconocimiento Facial EP2",
        "estado": "online",
        "docs": "/docs",
        "modelo_listo": MODELO is not None,
    }


@app.get("/health")
def estado():
    """Verifica que la API esté arriba y el modelo cargado."""
    return {
        "estado": "ok" if MODELO is not None else "modelo_no_cargado",
        "modelo_cargado": MODELO is not None,
        "clases": len(NOMBRES_CLASES),
        "umbral": UMBRAL_CONFIANZA,
        "personas_registradas": NOMBRES_CLASES[:10],
    }


@app.post("/predict", response_model=RespuestaPrediccion)
def predecir(entrada: EntradaImagen):
    """Clasifica la imagen: devuelve el nombre de la persona o 'no está en la base de datos'."""
    if MODELO is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado. Asegúrate de colocar modelo_cnn.keras y clases.json")
    matriz = extraer_matriz(entrada)
    try:
        # El modelo ya incluye la normalización (÷255), por eso se envían valores 0-255
        probabilidades = MODELO(matriz[np.newaxis].astype("float32"), training=False).numpy()[0]
    except Exception:
        registro.exception("Falló la inferencia")
        raise HTTPException(status_code=500, detail="Error interno al ejecutar el modelo")
    indice = int(np.argmax(probabilidades))
    confianza = float(probabilidades[indice])
    if confianza < UMBRAL_CONFIANZA:
        return RespuestaPrediccion(
            reconocido=False,
            nombre=None,
            confianza=confianza,
            mensaje="La persona no está en la base de datos",
        )
    nombre = NOMBRES_CLASES[indice]
    return RespuestaPrediccion(
        reconocido=True,
        nombre=nombre,
        confianza=confianza,
        mensaje=f"Persona reconocida: {nombre}",
    )
