# Guía de Despliegue - Dataset Preparation Notebooks

Esta guía te ayudará a desplegar la aplicación completa: el backend Django en Render y el frontend React/Next.js en Vercel.

---

## Estructura del Proyecto

\`\`\`
├── backend/              # Django REST API
│   ├── manage.py
│   ├── requirements.txt
│   ├── notebook_api/    # Configuración principal
│   └── notebooks/       # App de notebooks
├── app/                 # Frontend Next.js
├── components/          # Componentes React
└── DEPLOYMENT.md        # Esta guía
\`\`\`

---

## Parte 1: Desplegar Backend en Render

### Paso 1: Preparar el Repositorio

1. Sube tu código a GitHub (si aún no lo has hecho):
\`\`\`bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/tu-usuario/tu-repo.git
git push -u origin main
\`\`\`

### Paso 2: Configurar Render

1. Ve a [https://render.com](https://render.com) y crea una cuenta o inicia sesión

2. Haz clic en **"New +"** → **"Web Service"**

3. Conecta tu repositorio de GitHub

4. Configura el servicio con estos valores:

   - **Name**: `notebook-api` (o el nombre que prefieras)
   - **Environment**: `Python 3`
   - **Region**: Elige la más cercana a tu ubicación
   - **Branch**: `main`
   - **Root Directory**: `backend`
   - **Build Command**:
     \`\`\`bash
     pip install -r requirements.txt && python manage.py migrate && python manage.py collectstatic --noinput
     \`\`\`
   - **Start Command**:
     \`\`\`bash
     gunicorn notebook_api.wsgi:application
     \`\`\`

5. Haz clic en **"Advanced"** y agrega las variables de entorno:

   - `DJANGO_SECRET_KEY`: Genera una clave segura (puedes usar un generador online)
   - `DEBUG`: `False`
   - `PYTHON_VERSION`: `3.11.0`
   - `ALLOWED_HOSTS`: `.onrender.com`
   - `CORS_ALLOWED_ORIGINS`: `https://tu-app.vercel.app` (lo actualizaremos después)

6. Haz clic en **"Create Web Service"**

7. Espera a que el despliegue termine (puede tardar 5-10 minutos)

8. Una vez completado, copia la URL de tu API (algo como: `https://notebook-api.onrender.com`)

### Paso 3: Verificar el Backend

Visita estos endpoints para verificar que funciona:
- `https://tu-api.onrender.com/api/notebooks/list_all/`
- `https://tu-api.onrender.com/api/notebooks/notebook_07/detail/`

---

## Parte 2: Desplegar Frontend en Vercel

### Paso 1: Preparar Vercel CLI (Opcional)

Puedes desplegar desde la interfaz web o usando CLI:

\`\`\`bash
npm install -g vercel
\`\`\`

### Paso 2: Desplegar desde la Web

1. Ve a [https://vercel.com](https://vercel.com) e inicia sesión

2. Haz clic en **"Add New..."** → **"Project"**

3. Importa tu repositorio de GitHub

4. Vercel detectará automáticamente que es un proyecto Next.js

5. Configura las variables de entorno:
   - Haz clic en **"Environment Variables"**
   - Agrega: `NEXT_PUBLIC_API_URL` con valor `https://tu-api.onrender.com/api`
   - Asegúrate de usar la URL real de tu API de Render

6. Haz clic en **"Deploy"**

7. Espera a que el despliegue termine (2-3 minutos)

8. Copia tu URL de Vercel (algo como: `https://tu-proyecto.vercel.app`)

### Paso 3: Actualizar CORS en Render

1. Regresa a tu servicio en Render

2. Ve a **"Environment"**

3. Actualiza la variable `CORS_ALLOWED_ORIGINS`:
   \`\`\`
   https://tu-proyecto.vercel.app,https://tu-proyecto-git-main.vercel.app
   \`\`\`

4. Guarda los cambios (esto reiniciará el servicio)

### Paso 4: Verificar el Frontend

Visita tu URL de Vercel y verifica que:
- La página carga correctamente
- Puedes ver los tres notebooks en el menú lateral
- Al hacer clic en cada notebook, se carga su contenido
- Las pestañas (Secciones, Código, Puntos Clave) funcionan

---

## Parte 3: Desarrollo Local

### Backend Local

1. Navega a la carpeta backend:
\`\`\`bash
cd backend
\`\`\`

2. Crea un entorno virtual:
\`\`\`bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
\`\`\`

3. Instala las dependencias:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

4. Crea un archivo `.env` basado en `.env.example`:
\`\`\`bash
cp .env.example .env
\`\`\`

5. Edita `.env` con tus valores locales:
\`\`\`
DJANGO_SECRET_KEY=tu-clave-secreta-local
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ALLOWED_ORIGINS=http://localhost:3000
\`\`\`

6. Ejecuta las migraciones:
\`\`\`bash
python manage.py migrate
\`\`\`

7. Inicia el servidor:
\`\`\`bash
python manage.py runserver
\`\`\`

El backend estará disponible en: `http://localhost:8000`

### Frontend Local

1. Abre una nueva terminal en la raíz del proyecto

2. Instala las dependencias:
\`\`\`bash
npm install
\`\`\`

3. Crea un archivo `.env.local`:
\`\`\`bash
cp .env.local.example .env.local
\`\`\`

4. Edita `.env.local`:
\`\`\`
NEXT_PUBLIC_API_URL=http://localhost:8000/api
\`\`\`

5. Inicia el servidor de desarrollo:
\`\`\`bash
npm run dev
\`\`\`

El frontend estará disponible en: `http://localhost:3000`

---

## Solución de Problemas

### Error de CORS

**Síntoma**: El frontend no puede cargar datos del backend

**Solución**:
1. Verifica que `CORS_ALLOWED_ORIGINS` en Render incluya tu URL de Vercel
2. Asegúrate de incluir tanto la URL principal como la URL de preview
3. No incluyas barras finales en las URLs

### Error 500 en el Backend

**Síntoma**: Las peticiones al backend retornan error 500

**Solución**:
1. Revisa los logs en Render (pestaña "Logs")
2. Verifica que `DJANGO_SECRET_KEY` esté configurado
3. Confirma que las migraciones se ejecutaron correctamente

### El Frontend no se Despliega

**Síntoma**: El build falla en Vercel

**Solución**:
1. Verifica que no haya errores de TypeScript
2. Asegúrate de que todas las dependencias estén en `package.json`
3. Revisa los logs de build en Vercel

### Las Imágenes/Estilos no Cargan

**Síntoma**: El sitio se ve sin estilos después del despliegue

**Solución**:
1. Ejecuta `python manage.py collectstatic` en Render
2. Verifica que `whitenoise` esté instalado en requirements.txt
3. Confirma que `STATIC_ROOT` esté configurado en settings.py

---

## Comandos Útiles

### Backend (Django)

\`\`\`bash
# Crear migraciones
python manage.py makemigrations

# Aplicar migraciones
python manage.py migrate

# Crear superusuario
python manage.py createsuperuser

# Recolectar archivos estáticos
python manage.py collectstatic

# Ejecutar tests
python manage.py test
\`\`\`

### Frontend (Next.js)

\`\`\`bash
# Desarrollo
npm run dev

# Build de producción
npm run build

# Iniciar en producción
npm start

# Linting
npm run lint
\`\`\`

---

## Monitoreo y Mantenimiento

### Render

- **Logs**: Ve a tu servicio → pestaña "Logs"
- **Métricas**: Ve a "Metrics" para ver uso de CPU/memoria
- **Health checks**: Render hace ping automático a tu servicio

### Vercel

- **Analytics**: Habilita Vercel Analytics en tu proyecto
- **Logs**: Ve a tu deployment → "Logs" para ver errores
- **Preview Deployments**: Cada PR genera un preview automático

---

## Actualizaciones

### Backend

1. Haz cambios en tu código local
2. Commit y push a GitHub
3. Render desplegará automáticamente

### Frontend

1. Haz cambios en tu código local
2. Commit y push a GitHub
3. Vercel desplegará automáticamente

---

## Recursos Adicionales

- [Documentación de Render](https://render.com/docs)
- [Documentación de Vercel](https://vercel.com/docs)
- [Django Deployment Checklist](https://docs.djangoproject.com/en/5.0/howto/deployment/checklist/)
- [Next.js Deployment](https://nextjs.org/docs/deployment)

---

## Soporte

Si encuentras problemas:

1. Revisa los logs en Render y Vercel
2. Verifica que todas las variables de entorno estén configuradas
3. Confirma que las URLs de CORS sean correctas
4. Busca en Stack Overflow o la documentación oficial

---

¡Tu aplicación de visualización de notebooks está lista para producción! 🚀
