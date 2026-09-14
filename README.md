# 🔥 Terra-Fire: Predictive Wildfire Intelligence Engine

Terra-Fire es una plataforma de inteligencia predictiva y mapeo táctico de alta resolución diseñada para transformar la gestión de incendios forestales en México[cite: 1, 2]. El proyecto busca pasar de un modelo reactivo (basado en anomalías térmicas y contención) a un ecosistema preventivo, previendo la probabilidad de ignición a nivel micro-territorial entre 48 y 72 horas antes de que ocurra[cite: 1, 2].

## ✨ Características Principales

*   **Inteligencia Predictiva a 72 Horas:** Utiliza algoritmos de ensamblaje basados en árboles (XGBoost/LightGBM) entrenados con décadas de datos históricos para pronosticar el riesgo espacial de ignición[cite: 2].
*   **Arquitectura Offline-First (Last-Mile Delivery):** Una Progressive Web App (PWA) que almacena vectores en caché localmente (MapLibre, IndexedDB/SQLite vía WASM), permitiendo a brigadas comunitarias y ejidatarios consultar mapas de riesgo en sierras montañosas con nula conectividad celular[cite: 1, 2].
*   **Ingesta Satelital Automatizada:** Integración directa con APIs de Observación de la Tierra de acceso abierto (Copernicus Sentinel-2 y parámetros microclimáticos de ERA5-Land) para monitorear el estrés hídrico de la vegetación (NDMI, NBR) a resoluciones de 10-20 metros[cite: 1, 2].
*   **IA Explicable (XAI):** Implementación de valores SHAP para transparentar las decisiones del modelo (e.g., "Alto Déficit de Presión de Vapor + Pendiente Pronunciada"), construyendo confianza operativa en los despachadores y brigadistas[cite: 2].
*   **Interfaz de Alto Contraste (Operaciones Nocturnas):** UI diseñada para mitigar la fatiga visual con fondos azul marino y acentos en cyan y magenta, optimizando la visibilidad térmica en patrullajes nocturnos y ahorrando batería.

## 🛠️ Stack Tecnológico

**Backend & Datos:**
*   **Lenguaje:** Python
*   **Framework API:** Flask / Django
*   **Base de Datos Relacional:** PostgreSQL (con extensiones espaciales si es requerido)
*   **Infraestructura:** Dockerizado para un despliegue elástico (ideal para entornos como Google Cloud Run)

**Machine Learning & Ingesta:**
*   **Modelos:** XGBoost, LightGBM[cite: 1, 2]
*   **Métricas de Interpretabilidad:** SHAP[cite: 2]
*   **Fuentes Geoespaciales:** Sentinel-2 MSI, ERA5-Land Reanalysis[cite: 1, 2]

**Frontend (PWA):**
*   **Tecnología Base:** Progressive Web App (Offline-First)[cite: 1, 2]
*   **Renderizado de Mapas:** MapLibre GL para *vector tiles* ligeros[cite: 2]
*   **Caché en Cliente:** SQLite (WASM) / Service Workers[cite: 1, 2]

## 📂 Estructura del Repositorio

- `/data_pipeline`: Scripts automatizados para la extracción y preprocesamiento de índices satelitales (NDMI, NBR) y variables climáticas (VPD).
- `/ml_models`: Notebooks de investigación, entrenamiento del modelo espacial XGBoost y validación cruzada espacial (Blocked Spatial Cross-Validation).
- `/backend`: Lógica de la API (Python), manejo de variables de entorno y conexión a PostgreSQL.
- `/frontend_pwa`: Aplicación móvil responsiva optimizada para el almacenamiento offline de capas tácticas.
- `/database`: Migraciones y configuración del esquema inicial.

## 🚀 Despliegue Local (Desarrollo)

1. Clona este repositorio:
   ```bash
   git clone [https://github.com/tu-usuario/terra-fire.git](https://github.com/tu-usuario/terra-fire.git)
