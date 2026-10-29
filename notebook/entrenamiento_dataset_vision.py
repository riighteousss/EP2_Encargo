"""Entrenamiento de CNN conectada directamente al repositorio colaborativo:
https://github.com/The-Nuk/proyecto-dataset-vision

Este script:
1. Clona o lee el repositorio de GitHub.
2. Unifica alias y siglas (ej. 'mf 01' y 'Matías Figueroa', 'bp 01' y 'Benjamín Palma').
3. Separa 'compañero inf' / 'ci 01' como personas desconocidas para calibrar el umbral.
4. Aplica una división estratificada segura para evitar errores con pocas fotos.
5. Entrena las 12 configuraciones de Redes Neuronales Convolucionales (CNN).
6. Selecciona el mejor modelo según F1-Score Macro en validación.
7. Exporta 'modelo_cnn.keras' y 'clases.json' listos para la API (api/main.py) y la app móvil (iOS / Android).
"""

import csv
import json
import os
import shutil
import time
import unicodedata
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import callbacks, layers, models

# -------------------------------------------------------------
# 1. Configuración de Rutas y Parámetros
# -------------------------------------------------------------
SEMILLA = 42
TAMANO_IMAGEN = 128
EPOCAS_MAXIMAS = 40
TAMANO_LOTE = 16

REPO_GITHUB_URL = "https://github.com/The-Nuk/proyecto-dataset-vision.git"
CARPETA_REPO = os.path.abspath("proyecto-dataset-vision")
RUTA_IMAGENES = os.path.join(CARPETA_REPO, "data", "images")
RUTA_SUJETOS_CSV = os.path.join(CARPETA_REPO, "sujetos.csv")
RUTA_SALIDA = "resultados"

os.makedirs(f"{RUTA_SALIDA}/curvas", exist_ok=True)
os.makedirs(f"{RUTA_SALIDA}/modelos", exist_ok=True)
np.random.seed(SEMILLA)
tf.random.set_seed(SEMILLA)

print("Dispositivos GPU disponibles:", tf.config.list_physical_devices("GPU"))


# -------------------------------------------------------------
# 2. Funciones de Carga, Unificación de Alias y Normalización
# -------------------------------------------------------------
def normalizar_texto(texto):
    if not texto:
        return ""
    return unicodedata.normalize("NFC", str(texto)).strip()


MAPA_ALIAS = {
    "mf 01": "Matías Figueroa",
    "mf_01": "Matías Figueroa",
    "cv 01": "Carlos Vasquez",
    "cv_01": "Carlos Vasquez",
    "jm 01": "Javier Montupil",
    "jm_01": "Javier Montupil",
    "la 01": "Luciano Alcantara",
    "la_01": "Luciano Alcantara",
    "bg 01": "Benjamín Gálvez",
    "bg_01": "Benjamín Gálvez",
    "bp 01": "Benjamín Palma",
    "bp_01": "Benjamín Palma",
    "ci 01": "compañero inf",
    "ci_01": "compañero inf",
}


def asegurar_repositorio():
    if not os.path.exists(CARPETA_REPO):
        print(f"Clonando dataset desde {REPO_GITHUB_URL}...")
        os.system(f"git clone {REPO_GITHUB_URL} {CARPETA_REPO}")
    else:
        print(f"Repositorio ya presente en: {CARPETA_REPO}")


def leer_imagen_rgb(ruta):
    img = cv2.imread(ruta)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (TAMANO_IMAGEN, TAMANO_IMAGEN))


def cargar_dataset(ruta_imagenes, ruta_csv):
    mapa_csv = {}
    if os.path.exists(ruta_csv):
        with open(ruta_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for fila in reader:
                id_c = normalizar_texto(fila.get("id_carpeta", ""))
                nom = normalizar_texto(fila.get("nombre_original") or fila.get("nombre_normalizado", ""))
                if id_c and nom:
                    mapa_csv[id_c.lower()] = nom

    imagenes_conocidos, etiquetas_conocidos = [], []
    fotos_desconocidas = []

    if not os.path.exists(ruta_imagenes):
        raise FileNotFoundError(f"No se encontró la carpeta de imágenes en: {ruta_imagenes}")

    for carpeta_id in sorted(os.listdir(ruta_imagenes)):
        ruta_carpeta = os.path.join(ruta_imagenes, carpeta_id)
        if not os.path.isdir(ruta_carpeta) or carpeta_id.startswith("."):
            continue

        c_norm = normalizar_texto(carpeta_id)
        c_lower = c_norm.lower()

        nombre_final = mapa_csv.get(c_lower, MAPA_ALIAS.get(c_lower, c_norm))
        nombre_final = normalizar_texto(nombre_final)

        es_desconocido = any(p in c_lower or p in nombre_final.lower()
                             for p in ["compañero", "companero", "ci 01", "ci_01", "desconocid"])

        for archivo in os.listdir(ruta_carpeta):
            if archivo.lower().endswith((".jpg", ".jpeg", ".png")):
                img = leer_imagen_rgb(os.path.join(ruta_carpeta, archivo))
                if img is not None:
                    if es_desconocido:
                        fotos_desconocidas.append(img)
                    else:
                        imagenes_conocidos.append(img)
                        etiquetas_conocidos.append(nombre_final)

    return (np.array(imagenes_conocidos, dtype="uint8"),
            np.array(etiquetas_conocidos),
            np.array(fotos_desconocidas, dtype="uint8") if len(fotos_desconocidas) > 0 else np.array([]))


def division_estratificada_segura(X_data, y_data, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """Garantiza que toda clase esté en train/val/test sin error de pocos miembros."""
    rng = np.random.RandomState(random_seed)
    idx_train, idx_val, idx_test = [], [], []

    for clase_id in np.unique(y_data):
        indices_clase = np.where(y_data == clase_id)[0]
        rng.shuffle(indices_clase)
        n = len(indices_clase)

        if n <= 2:
            idx_train.append(indices_clase[0])
            idx_val.append(indices_clase[-1])
            idx_test.append(indices_clase[-1])
        elif n == 3:
            idx_train.append(indices_clase[0])
            idx_val.append(indices_clase[1])
            idx_test.append(indices_clase[2])
        else:
            n_val = max(1, int(round(n * val_ratio)))
            n_test = max(1, int(round(n * test_ratio)))
            if n_val + n_test >= n:
                n_val = 1
                n_test = 1
            n_train = n - n_val - n_test
            idx_train.extend(indices_clase[:n_train])
            idx_val.extend(indices_clase[n_train:n_train + n_val])
            idx_test.extend(indices_clase[n_train + n_val:])

    idx_tr = np.array(idx_train)
    idx_v = np.array(idx_val)
    idx_te = np.array(idx_test)

    return X_data[idx_tr], X_data[idx_v], X_data[idx_te], y_data[idx_tr], y_data[idx_v], y_data[idx_te]


# -------------------------------------------------------------
# 3. Construcción de Modelos CNN
# -------------------------------------------------------------
def construir_cnn(numero_clases, numero_bloques, usar_dropout, usar_batchnorm, usar_aumento):
    entrada = layers.Input(shape=(TAMANO_IMAGEN, TAMANO_IMAGEN, 3))
    x = entrada
    if usar_aumento:
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(15 / 360)(x)
        x = layers.RandomZoom(0.1)(x)
        x = layers.RandomBrightness(0.2, value_range=(0, 255))(x)
    x = layers.Rescaling(1.0 / 255)(x)

    for filtros in [32, 64, 128][:numero_bloques]:
        x = layers.Conv2D(filtros, 3, padding="same")(x)
        if usar_batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    if usar_dropout:
        x = layers.Dropout(0.4)(x)
    salida = layers.Dense(numero_clases, activation="softmax")(x)
    return models.Model(entrada, salida)


# -------------------------------------------------------------
# 4. Flujo Principal de Ejecución
# -------------------------------------------------------------
def main():
    asegurar_repositorio()
    print("Cargando y unificando dataset...")
    X, etiquetas_texto, fotos_desconocidas = cargar_dataset(RUTA_IMAGENES, RUTA_SUJETOS_CSV)
    print(f"Total imágenes de alumnos conocidos: {len(X)}")
    print(f"Total fotos de personas desconocidas: {len(fotos_desconocidas)}")

    codificador = LabelEncoder()
    y = codificador.fit_transform(etiquetas_texto)
    nombres_clases = list(codificador.classes_)
    numero_clases = len(nombres_clases)

    conteo = pd.Series(etiquetas_texto).value_counts()
    print("\nDistribución unificada de fotos por alumno:")
    print(conteo)

    # División estratificada segura
    X_train, X_val, X_test, y_train, y_val, y_test = division_estratificada_segura(
        X, y, val_ratio=0.15, test_ratio=0.15, random_seed=SEMILLA
    )
    print(f"\nConjuntos: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

    arquitecturas = {"CNN-A": 1, "CNN-B": 2, "CNN-C": 3}
    regularizaciones = {
        "Base": dict(usar_dropout=False, usar_batchnorm=False, usar_aumento=False),
        "+Dropout": dict(usar_dropout=True, usar_batchnorm=False, usar_aumento=False),
        "+BN+Dropout": dict(usar_dropout=True, usar_batchnorm=True, usar_aumento=False),
        "+Aug+BN+Dropout": dict(usar_dropout=True, usar_batchnorm=True, usar_aumento=True),
    }

    configuraciones = []
    for arch_nombre, bloques in arquitecturas.items():
        for reg_nombre, opciones in regularizaciones.items():
            configuraciones.append(dict(
                id=f"C{len(configuraciones)+1}",
                arquitectura=arch_nombre,
                regularizacion=reg_nombre,
                numero_bloques=bloques,
                **opciones
            ))

    print("\nEntrenando 12 configuraciones...")
    resultados, historiales = [], {}
    for cfg in configuraciones:
        tf.keras.backend.clear_session()
        modelo = construir_cnn(
            numero_clases,
            cfg["numero_bloques"],
            cfg["usar_dropout"],
            cfg["usar_batchnorm"],
            cfg["usar_aumento"]
        )
        modelo.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        control = [
            callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
        ]

        inicio = time.time()
        historia = modelo.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCAS_MAXIMAS,
            batch_size=TAMANO_LOTE,
            callbacks=control,
            verbose=0
        )
        h = historia.history
        mejor_epoca = int(np.argmin(h["val_loss"]))
        y_pred_val = np.argmax(modelo.predict(X_val, verbose=0), axis=1)

        val_acc = accuracy_score(y_val, y_pred_val)
        val_f1 = f1_score(y_val, y_pred_val, average="macro", zero_division=0)
        resultados.append({
            "id": cfg["id"],
            "arquitectura": cfg["arquitectura"],
            "regularizacion": cfg["regularizacion"],
            "val_accuracy": val_acc,
            "val_f1_macro": val_f1,
            "val_loss": h["val_loss"][mejor_epoca],
            "epocas": len(h["loss"]),
            "tiempo_seg": round(time.time() - inicio, 1),
            "parametros": modelo.count_params()
        })
        historiales[cfg["id"]] = h
        modelo.save(f"{RUTA_SALIDA}/modelos/{cfg['id']}.keras")
        print(f"[{cfg['id']}] {cfg['arquitectura']} {cfg['regularizacion']} -> Val Acc: {val_acc:.3f} | Val F1: {val_f1:.3f}")

    tabla = pd.DataFrame(resultados)
    tabla.to_csv(f"{RUTA_SALIDA}/resultados_12_configuraciones.csv", index=False)

    mejor = tabla.sort_values(["val_f1_macro", "val_loss"], ascending=[False, True]).iloc[0]
    print(f"\nMejor modelo seleccionado: {mejor['id']} ({mejor['arquitectura']} {mejor['regularizacion']})")

    modelo_final = tf.keras.models.load_model(f"{RUTA_SALIDA}/modelos/{mejor['id']}.keras")

    y_pred_test = np.argmax(modelo_final.predict(X_test, verbose=0), axis=1)
    acc_test = accuracy_score(y_test, y_pred_test)
    print(f"Accuracy en Test: {acc_test:.3f}")

    # Calibración del umbral
    confianza_validacion = modelo_final.predict(X_val, verbose=0)
    pred_val = confianza_validacion.argmax(axis=1)
    confianza_aciertos_val = confianza_validacion.max(axis=1)[pred_val == y_val]

    if len(fotos_desconocidas) > 0:
        confianza_desc = modelo_final.predict(fotos_desconocidas, verbose=0).max(axis=1)
        umbrales = np.arange(0.30, 1.00, 0.01)
        puntajes = [((confianza_aciertos_val >= u).mean() + (confianza_desc < u).mean()) / 2 for u in umbrales]
        umbral_confianza = float(umbrales[int(np.argmax(puntajes))])
        print(f"Umbral calibrado con fotos desconocidas: {round(umbral_confianza, 2)}")
    else:
        umbral_confianza = float(max(0.5, np.percentile(confianza_aciertos_val, 10)))
        print(f"Umbral por percentil 10: {round(umbral_confianza, 2)}")

    # Exportar modelo y metadatos
    modelo_final.save(f"{RUTA_SALIDA}/modelo_cnn.keras")
    with open(f"{RUTA_SALIDA}/clases.json", "w", encoding="utf-8") as f:
        json.dump({
            "clases": nombres_clases,
            "umbral": umbral_confianza,
            "tamano_imagen": TAMANO_IMAGEN,
            "configuracion": f"{mejor['id']} {mejor['arquitectura']} {mejor['regularizacion']}"
        }, f, ensure_ascii=False, indent=2)

    # Copiar a carpeta api/ si existe
    if os.path.exists("../api"):
        shutil.copy(f"{RUTA_SALIDA}/modelo_cnn.keras", "../api/modelo_cnn.keras")
        shutil.copy(f"{RUTA_SALIDA}/clases.json", "../api/clases.json")
        print("Archivos copiados automáticamente a api/")

    print("\nEntrenamiento finalizado con éxito.")


if __name__ == "__main__":
    main()
