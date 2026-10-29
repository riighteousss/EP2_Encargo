# EP2 — AVY1103 Deep Learning: Reconocimiento Facial (API + App Multiplataforma iOS / Android)

Aplicación de Visión Artificial que captura una fotografía (cámara o galería), la convierte a una matriz RGB 128×128×3 (49.152 bytes) y la envía a un backend FastAPI con una Red Neuronal Convolucional (CNN). La API clasifica el rostro y responde con el nombre de la persona o indicando que no está registrada en la base de datos.

---

## 📱 ¿Esta aplicación sirve para iPhone (iOS)?

**Sí, es 100% compatible tanto con iPhone (iOS) como con Android.**

Para que funcione en iPhone se implementaron los siguientes ajustes necesarios por las políticas de Apple:
1. **Permisos de Privacidad en iOS (`ios/Runner/Info.plist`)**:
   - `NSCameraUsageDescription`: Autorización para acceder a la cámara del iPhone.
   - `NSPhotoLibraryUsageDescription`: Autorización para seleccionar fotos desde la app Fotos/Galería.
   - `NSMicrophoneUsageDescription`: Autorización de audio requerida por la API de cámara.
2. **Seguridad de Red (App Transport Security - ATS)**:
   - Configurado en `Info.plist` para permitir conexiones HTTP de desarrollo a servidores en red local (ej. `http://192.168.x.x:8000`) o en la nube (`https://...`).
3. **Selector de Fuente de Imagen**:
   - Ahora incluye botón de **Cámara** y botón de **Galería** (esencial si pruebas en el Simulador de iPhone en Mac donde no hay cámara de hardware).
4. **Configurador de URL en Vivo**:
   - Toca el icono de configuración (o el chip de la URL en la barra superior) para escribir o cambiar la URL de la API sin tener que recompilar la aplicación.

> **Nota para compilar en iPhone**:
> Apple exige **macOS con Xcode** para generar el archivo instalable de iOS (`.ipa`) o para instalar la app vía cable USB.
> - Si tienes Mac: Ejecuta `flutter run -d ios` o conecta tu iPhone y ábrelo en Xcode.
> - Si estás en Windows: Puedes probar la app en **Android** (`flutter run` / `flutter build apk`) o ejecutar la app en **Flutter Web** (`flutter run -d chrome`), o usar servicios de compilación en la nube (como Codemagic o GitHub Actions).

---

## 🔗 Conexión con el Dataset: `proyecto-dataset-vision`

El dataset colaborativo se encuentra en:
👉 **[https://github.com/The-Nuk/proyecto-dataset-vision](https://github.com/The-Nuk/proyecto-dataset-vision)**

### ¿Cómo se conecta la app con el dataset?

Una aplicación móvil **no puede conectarse en vivo a un repositorio de Git/DVC para comparar fotos pixel por pixel** en tiempo real. La arquitectura de Deep Learning sigue un flujo de 3 etapas:

```
[1. Dataset Colaborativo Git + DVC]
   GitHub: The-Nuk/proyecto-dataset-vision
   - data/images/ (persona001, persona002, ...)
   - sujetos.csv (persona001 -> Nombre Real)
                   │
                   ▼
[2. Entrenamiento en Google Colab / Notebook]
   - Clona el repositorio y descarga imágenes con DVC.
   - Mapea carpetas a nombres reales con 'sujetos.csv'.
   - Entrena y evalúa 12 configuraciones de CNN.
   - Genera: 'modelo_cnn.keras' y 'clases.json'.
                   │
                   ▼
[3. API Backend (FastAPI / Render / Local)]
   - Carga 'modelo_cnn.keras' y 'clases.json'.
   - Expone endpoints: GET /health y POST /predict.
                   ▲
                   │ Envia matriz RGB 128x128x3
                   │ Recibe: { "reconocido": true, "nombre": "..." }
                   │
[4. App Móvil Flutter (iPhone & Android)]
   - Toma foto -> resize 128x128 -> POST /predict -> muestra resultado en pantalla.
```

---

## 📂 Estructura del Proyecto

| Carpeta / Archivo | Descripción |
|---|---|
| `notebook/entrenamiento_dataset_vision.py` | Script que clona `proyecto-dataset-vision`, descarga datos DVC, mapea `sujetos.csv` y entrena las 12 CNN |
| `notebook/EP2_Entrenamiento_CNN.ipynb` | Cuaderno Jupyter para ejecutar en Google Colab con GPU |
| `api/main.py` | Backend FastAPI con CORS, carga de modelo y mapeo de `sujetos.csv` |
| `api/requirements.txt` | Dependencias del backend (`fastapi`, `uvicorn`, `tensorflow-cpu`, `numpy`) |
| `api/Dockerfile` | Contenedor Docker para despliegue en la nube (Render, Railway, etc.) |
| `app/lib/main.dart` | App Flutter con soporte para Cámara + Galería, selector de URL y estado de API |
| `app/pubspec.yaml` | Dependencias Flutter (`http`, `image_picker`, `image`, `cupertino_icons`) |
| `app/ios/Runner/Info.plist` | Configuración y permisos para iPhone (cámara, galería, ATS) |
| `app/android/app/src/main/AndroidManifest.xml` | Permisos de Android (cámara, almacenamiento, internet) |

---

## 🚀 Guía Paso a Paso

### Paso 1: Entrenar el Modelo con el Dataset de GitHub

1. Abre **Google Colab** y activa GPU T4 (`Entorno de ejecución` → `Cambiar tipo de entorno de ejecución` → `GPU`).
2. Sube el notebook `notebook/EP2_Entrenamiento_CNN.ipynb` o ejecuta una celda para clonar el repositorio:
   ```bash
   !git clone https://github.com/The-Nuk/proyecto-dataset-vision.git /content/proyecto-dataset-vision
   ```
3. Descarga las imágenes con DVC en Colab:
   ```bash
   !pip install dvc dvc-gdrive
   %cd /content/proyecto-dataset-vision
   !dvc remote modify --local almacenamiento gdrive_client_id "TU_ID_CLIENTE_GCP"
   !dvc remote modify --local almacenamiento gdrive_client_secret "TU_SECRET_GCP"
   !dvc pull
   %cd /content
   ```
   *(Alternativa: Si las carpetas de fotos ya están en tu Google Drive, monta Drive con `from google.colab import drive; drive.mount('/content/drive')` y apunta la ruta a dicha carpeta).*
4. Ejecuta el script `notebook/entrenamiento_dataset_vision.py` o las celdas del notebook.
5. Al finalizar se generarán:
   - `modelo_cnn.keras`: Pesos de la mejor red neuronal.
   - `clases.json`: Lista de personas ordenadas y umbral de confianza calibrado.
6. Descárgalos y colócalos dentro de la carpeta `api/`.

---

### Paso 2: Ejecutar la API (FastAPI)

1. Abre tu terminal en la carpeta `api/`:
   ```bash
   cd api
   pip install -r requirements.txt
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   & "C:\Users\lolit\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
2. Abre tu navegador en `http://localhost:8000/docs` para ver y probar la documentación Swagger interactiva.
3. Para desplegar en la nube gratis:
   - Sube la carpeta `api/` (con `modelo_cnn.keras`, `clases.json`, `main.py`, `Dockerfile`) a GitHub.
   - Crea un **Web Service** en [Render.com](https://render.com) seleccionando el entorno **Docker**.
   - Obtendrás una URL pública segura (ej: `https://mi-reconocimiento.onrender.com`).

---

### Paso 3: Ejecutar la App Móvil

Dentro de la carpeta `app/`:

#### En Android:
```bash
cd app
flutter run
```
Para generar el instalador APK:
```bash
flutter build apk --release
```
El archivo queda en: `build/app/outputs/flutter-apk/app-release.apk`.

#### En iPhone (iOS):
1. Asegúrate de tener los pods instalados:
   ```bash
   cd app/ios
   pod install
   cd ..
   ```
2. Ejecuta en el simulador o dispositivo físico conectado:
   ```bash
   flutter run -d ios
   ```

---

## 📡 Cómo conectar la App Móvil con la API según el entorno

| Entorno de la App | URL a ingresar en la app |
|---|---|
| **Simulador de iPhone (iOS Simulator)** | `http://127.0.0.1:8000` |
| **iPhone Físico (conectado al mismo Wi-Fi que tu PC)** | `http://192.168.X.X:8000` *(IP local de tu PC)* |
| **Emulador de Android** | `http://10.0.2.2:8000` |
| **Android Físico (conectado al mismo Wi-Fi)** | `http://192.168.X.X:8000` |
| **Cualquier teléfono fuera de casa / Nube** | `https://tu-api.onrender.com` *(o túnel ngrok)* |

> Puedes cambiar la URL directamente desde el botón **Ajustes (⚙️)** de la app sin necesidad de recompilar el código.

---

## 🔬 Especificación Técnica del Modelo
- **Entrada:** Matriz `128×128×3` (RGB uint8, valores entre 0 y 255).
- **Preprocesamiento interno:** Capa `Rescaling(1./255)` integrada en la CNN (evita discrepancias de normalización).
- **Data Augmentation integrado:** Volteo horizontal, rotación aleatoria de ±15°, zoom de 10% y variación de brillo (solo activo en entrenamiento).
- **Métrica de selección:** F1-Score Macro en el conjunto de validación.
- **Detección de desconocidos:** Si la probabilidad máxima predicha es menor que el `umbral` (por defecto `0.60`), la API responde `"reconocido": false`.
