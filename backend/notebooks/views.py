from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Notebook
from .serializers import NotebookSerializer, NotebookDetailSerializer


class NotebookViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for viewing notebooks.
    """
    queryset = Notebook.objects.all()
    serializer_class = NotebookSerializer

    @action(detail=False, methods=['get'], url_path='(?P<notebook_id>[^/.]+)/detail')
    def get_detail(self, request, notebook_id=None):
        """Get detailed information about a specific notebook"""
        
        notebook_data = {
            'notebook_07': {
                'notebook_id': 'notebook_07',
                'title': 'Division del Dataset',
                'description': 'Notebook que divide el dataset NSL-KDD en conjuntos de entrenamiento, validacion y prueba usando estratificacion.',
                'sections': [
                    {
                        'title': 'Objetivo',
                        'content': 'Dividir el dataset en tres conjuntos: 60% entrenamiento, 20% validacion, 20% prueba, manteniendo las proporciones de protocol_type.'
                    },
                    {
                        'title': 'Tecnicas Implementadas',
                        'content': 'Utilizacion de train_test_split de sklearn con estratificacion sobre la columna protocol_type para mantener la distribucion de protocolos (tcp, udp, icmp) en todos los conjuntos.'
                    },
                    {
                        'title': 'Funcion Principal',
                        'content': 'train_val_test_split() - Funcion reutilizable que acepta parametros configurables para las proporciones y la columna de estratificacion.'
                    },
                ],
                'code_examples': [
                    {
                        'title': 'Division Estratificada',
                        'code': '''from sklearn.model_selection import train_test_split

def train_val_test_split(data, strat_col, test_size=0.2, val_size=0.2, random_state=42):
    # Primera division: train+val vs test
    train_val, test = train_test_split(
        data, 
        test_size=test_size, 
        stratify=data[strat_col],
        random_state=random_state
    )
    
    # Segunda division: train vs val
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        stratify=train_val[strat_col],
        random_state=random_state
    )
    
    return train, val, test''',
                        'explanation': 'Funcion que realiza dos divisiones sucesivas manteniendo la estratificacion en ambas etapas.'
                    },
                ],
                'key_points': [
                    'Dataset con 125,973 registros y 42 columnas',
                    'Uso de random_state=42 para reproducibilidad',
                    'Estratificacion sobre protocol_type para mantener proporciones',
                    'Division: 60% training / 20% validation / 20% test',
                    'Visualizaciones con histogramas para verificar estratificacion'
                ]
            },
            'notebook_08': {
                'notebook_id': 'notebook_08',
                'title': 'Preparacion del Dataset',
                'description': 'Notebook que implementa tecnicas de limpieza y transformacion de datos para preparar el dataset NSL-KDD para Machine Learning.',
                'sections': [
                    {
                        'title': 'Manejo de Valores Nulos',
                        'content': 'Tres metodos implementados: eliminacion de filas con NaN, eliminacion de columnas con NaN, e imputacion usando SimpleImputer con estrategia de mediana.'
                    },
                    {
                        'title': 'Conversion de Categoricos',
                        'content': 'Tres enfoques: OrdinalEncoder (sklearn), OneHotEncoder (sklearn) para matrices sparse, y pd.get_dummies() (pandas) para conversion mas sencilla.'
                    },
                    {
                        'title': 'Escalado de Features',
                        'content': 'Implementacion de normalizacion y estandarizacion, con cuidado especial de NO aplicar sobre las etiquetas (labels), solo sobre las features.'
                    },
                ],
                'code_examples': [
                    {
                        'title': 'Imputacion con SimpleImputer',
                        'code': '''from sklearn.impute import SimpleImputer

# Separar features y labels
X = data.drop('label', axis=1)
y = data['label']

# Imputar valores faltantes con la mediana
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)''',
                        'explanation': 'SimpleImputer reemplaza valores faltantes con la mediana de cada columna numerica.'
                    },
                    {
                        'title': 'One-Hot Encoding',
                        'code': '''from sklearn.preprocessing import OneHotEncoder

# Identificar columnas categoricas
cat_columns = ['protocol_type', 'service', 'flag']

# Aplicar one-hot encoding
encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[cat_columns])''',
                        'explanation': 'OneHotEncoder convierte variables categoricas en vectores binarios, creando una columna por cada categoria.'
                    },
                ],
                'key_points': [
                    'Separacion correcta entre X (features) e y (labels)',
                    'OrdinalEncoder puede crear relaciones ordinales falsas',
                    'OneHotEncoder preferido para categoricos sin orden',
                    'Escalado NO se aplica sobre etiquetas',
                    'SimpleImputer con mediana es robusto a outliers'
                ]
            },
            'notebook_09': {
                'notebook_id': 'notebook_09',
                'title': 'Transformadores y Pipelines Personalizados',
                'description': 'Notebook que crea componentes reutilizables para preprocesamiento usando transformadores personalizados y pipelines de sklearn.',
                'sections': [
                    {
                        'title': 'Transformadores Personalizados',
                        'content': 'Creacion de clases que heredan de BaseEstimator y TransformerMixin para implementar transformaciones personalizadas compatibles con pipelines de sklearn.'
                    },
                    {
                        'title': 'Pipeline Numerico',
                        'content': 'Pipeline que combina imputacion de valores faltantes con SimpleImputer y escalado robusto con RobustScaler para manejar outliers.'
                    },
                    {
                        'title': 'ColumnTransformer',
                        'content': 'Uso de ColumnTransformer para aplicar diferentes transformaciones a columnas numericas y categoricas de forma simultanea y organizada.'
                    },
                ],
                'code_examples': [
                    {
                        'title': 'Transformador Personalizado',
                        'code': '''from sklearn.base import BaseEstimator, TransformerMixin

class DeleteNanRows(BaseEstimator, TransformerMixin):
    """Elimina filas con valores NaN"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.dropna()''',
                        'explanation': 'Transformador personalizado que elimina filas con valores faltantes, compatible con pipelines de sklearn.'
                    },
                    {
                        'title': 'Pipeline Completo',
                        'code': '''from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Pipeline para columnas numericas
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('rbst_scaler', RobustScaler()),
])

# Pipeline completo con ColumnTransformer
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

# Aplicar el pipeline
X_prepared = full_pipeline.fit_transform(X)''',
                        'explanation': 'Pipeline completo que aplica diferentes transformaciones a columnas numericas y categoricas en un solo paso.'
                    },
                ],
                'key_points': [
                    'Transformadores personalizados extienden BaseEstimator y TransformerMixin',
                    'Pipelines automatizan el preprocesamiento de datos',
                    'ColumnTransformer permite transformaciones diferentes por tipo de columna',
                    'RobustScaler es mas resistente a outliers que StandardScaler',
                    'Los pipelines garantizan reproducibilidad y evitan data leakage'
                ]
            },
        }
        
        if notebook_id not in notebook_data:
            return Response(
                {'error': 'Notebook no encontrado'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        serializer = NotebookDetailSerializer(data=notebook_data[notebook_id])
        if serializer.is_valid():
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['get'])
    def list_all(self, request):
        """List all available notebooks"""
        notebooks = [
            {
                'id': 1,
                'notebook_id': 'notebook_07',
                'title': 'Division del Dataset',
                'description': 'Estratificacion y division del dataset NSL-KDD',
                'order': 1
            },
            {
                'id': 2,
                'notebook_id': 'notebook_08',
                'title': 'Preparacion del Dataset',
                'description': 'Limpieza y transformacion de datos',
                'order': 2
            },
            {
                'id': 3,
                'notebook_id': 'notebook_09',
                'title': 'Transformadores y Pipelines',
                'description': 'Componentes reutilizables para preprocesamiento',
                'order': 3
            },
        ]
        return Response(notebooks)
