# Dataset Preparation Notebooks Viewer

Aplicación web full-stack para visualizar y explicar notebooks de preparación de datasets utilizando el dataset NSL-KDD (Network Security Laboratory - Knowledge Discovery in Databases).

## Características

- **Backend Django REST API**: API robusta que sirve información detallada sobre los notebooks
- **Frontend React/Next.js**: Interfaz moderna e interactiva para explorar los notebooks
- **3 Notebooks Incluidos**:
  1. **División del Dataset**: Estratificación y división en train/val/test
  2. **Preparación del Dataset**: Limpieza y transformación de datos
  3. **Transformadores y Pipelines**: Componentes reutilizables para ML

## Vista Previa

La aplicación incluye:
- Menú lateral con navegación entre notebooks
- Pestañas para organizar contenido: Secciones, Código, Puntos Clave
- Bloques de código con syntax highlighting y botón de copiar
- Diseño responsivo con soporte para modo oscuro
- Explicaciones detalladas de cada técnica implementada

## Tecnologías Utilizadas

### Backend
- Django 5.0.1
- Django REST Framework 3.14.0
- Django CORS Headers
- Gunicorn
- WhiteNoise

### Frontend
- Next.js 16
- React 19
- TypeScript
- Tailwind CSS v4
- shadcn/ui components
- Lucide React (iconos)

## Estructura del Proyecto

\`\`\`
├── backend/
│   ├── notebook_api/       # Configuración Django
│   ├── notebooks/          # App de notebooks con API
│   ├── manage.py
│   └── requirements.txt
├── app/                    # Páginas Next.js
├── components/             # Componentes React
│   ├── notebook-viewer.tsx
│   ├── notebook-menu.tsx
│   └── code-block.tsx
├── lib/                    # Utilidades
└── public/                 # Assets estáticos
\`\`\`

## Instalación Local

### Requisitos Previos
- Python 3.11+
- Node.js 18+
- npm o yarn

### Backend

1. Navega al directorio del backend:
\`\`\`bash
cd backend
\`\`\`

2. Crea y activa un entorno virtual:
\`\`\`bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
\`\`\`

3. Instala las dependencias:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

4. Configura las variables de entorno:
\`\`\`bash
cp .env.example .env
# Edita .env con tus valores
\`\`\`

5. Ejecuta las migraciones:
\`\`\`bash
python manage.py migrate
\`\`\`

6. Inicia el servidor:
\`\`\`bash
python manage.py runserver
\`\`\`

El backend estará disponible en `http://localhost:8000`

### Frontend

1. En la raíz del proyecto, instala las dependencias:
\`\`\`bash
npm install
\`\`\`

2. Configura las variables de entorno:
\`\`\`bash
cp .env.local.example .env.local
# Edita .env.local
\`\`\`

3. Inicia el servidor de desarrollo:
\`\`\`bash
npm run dev
\`\`\`

El frontend estará disponible en `http://localhost:3000`

## Despliegue

Consulta la [Guía de Despliegue](DEPLOYMENT.md) para instrucciones detalladas sobre cómo desplegar en Render (backend) y Vercel (frontend).

### Resumen Rápido

**Backend en Render:**
1. Conecta tu repositorio de GitHub
2. Configura el servicio con Python 3
3. Establece las variables de entorno
4. Despliega

**Frontend en Vercel:**
1. Importa tu repositorio
2. Configura `NEXT_PUBLIC_API_URL`
3. Despliega
4. Actualiza CORS en el backend

## API Endpoints

### Listar todos los notebooks
\`\`\`
GET /api/notebooks/list_all/
\`\`\`

Respuesta:
\`\`\`json
[
  {
    "id": 1,
    "notebook_id": "notebook_07",
    "title": "División del Dataset",
    "description": "...",
    "order": 1
  }
]
\`\`\`

### Obtener detalles de un notebook
\`\`\`
GET /api/notebooks/{notebook_id}/detail/
\`\`\`

Respuesta:
\`\`\`json
{
  "notebook_id": "notebook_07",
  "title": "División del Dataset",
  "description": "...",
  "sections": [...],
  "code_examples": [...],
  "key_points": [...]
}
\`\`\`

## Notebooks Incluidos

### 1. División del Dataset (Notebook 07)
- Estratificación sobre `protocol_type`
- División 60/20/20 (train/val/test)
- Función reutilizable `train_val_test_split()`
- Visualizaciones para verificar distribución

### 2. Preparación del Dataset (Notebook 08)
- Manejo de valores nulos (3 métodos)
- Conversión de variables categóricas
- Escalado de features
- Separación correcta X/y

### 3. Transformadores y Pipelines (Notebook 09)
- Transformadores personalizados
- Pipeline numérico con RobustScaler
- ColumnTransformer para diferentes tipos
- Automatización del preprocesamiento

## Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Haz fork del repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## Contacto

Para preguntas o soporte, por favor abre un issue en el repositorio.

---

Desarrollado con Django, Next.js y mucho café.
