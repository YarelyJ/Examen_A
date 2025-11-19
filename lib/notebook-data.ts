export interface Section {
  title: string
  content: string
}

export interface CodeExample {
  title: string
  code: string
  explanation: string
}

export interface DatasetStats {
  totalRecords: number
  totalFeatures: number
  numericFeatures: number
  categoricalFeatures: number
  missingValues: number
  classDistribution: { label: string; count: number; percentage: number }[]
}

export interface MethodComparison {
  method: string
  description: string
  advantages: string[]
  disadvantages: string[]
  useCase: string
}

export interface NotebookDetail {
  notebook_id: string
  title: string
  description: string
  sections: Section[]
  code_examples: CodeExample[]
  key_points: string[]
  dataset_stats?: DatasetStats
  method_comparisons?: MethodComparison[]
  technical_details?: {
    algorithms: string[]
    complexity: string
    prerequisites: string[]
  }
}

export interface NotebookItem {
  id: number
  notebook_id: string
  title: string
  description: string
  order: number
}

export const notebooksData: NotebookItem[] = [
  {
    id: 1,
    notebook_id: 'notebook_07',
    title: '07 - División del Dataset',
    description: 'División estratificada del dataset en entrenamiento, validación y prueba',
    order: 1,
  },
  {
    id: 2,
    notebook_id: 'notebook_08',
    title: '08 - Preparación del Dataset',
    description: 'Limpieza, transformación y encoding de datos para Machine Learning',
    order: 2,
  },
  {
    id: 3,
    notebook_id: 'notebook_09',
    title: '09 - Transformadores y Pipelines',
    description: 'Creación de transformadores personalizados y pipelines reutilizables',
    order: 3,
  },
]

export const notebookDetails: Record<string, NotebookDetail> = {
  notebook_07: {
    notebook_id: 'notebook_07',
    title: '07 - División del Dataset',
    description: 'División estratificada del dataset NSL-KDD en conjuntos de entrenamiento, validación y prueba',
    dataset_stats: {
      totalRecords: 125973,
      totalFeatures: 42,
      numericFeatures: 34,
      categoricalFeatures: 8,
      missingValues: 0,
      classDistribution: [
        { label: 'normal', count: 67343, percentage: 53.5 },
        { label: 'anomaly', count: 58630, percentage: 46.5 },
      ],
    },
    technical_details: {
      algorithms: ['train_test_split', 'Stratified Sampling'],
      complexity: 'O(n log n)',
      prerequisites: ['pandas', 'scikit-learn', 'numpy'],
    },
    method_comparisons: [
      {
        method: 'Random Split',
        description: 'División aleatoria sin considerar distribución de clases',
        advantages: ['Simple de implementar', 'Rápido', 'No requiere conocimiento previo de datos'],
        disadvantages: ['Puede crear conjuntos no representativos', 'Riesgo de sampling bias', 'Distribución desigual de clases'],
        useCase: 'Datasets muy grandes y balanceados',
      },
      {
        method: 'Stratified Split (Utilizado)',
        description: 'División manteniendo proporciones de clases/categorías',
        advantages: ['Mantiene distribución proporcional', 'Reduce sampling bias', 'Conjuntos representativos'],
        disadvantages: ['Requiere definir columna de estratificación', 'Ligeramente más lento', 'No funciona con múltiples columnas simultáneas'],
        useCase: 'Datasets desbalanceados o con categorías importantes',
      },
      {
        method: 'K-Fold Cross-Validation',
        description: 'División en K subconjuntos para validación cruzada',
        advantages: ['Mejor estimación del rendimiento', 'Usa todos los datos', 'Reduce varianza de evaluación'],
        disadvantages: ['K veces más costoso computacionalmente', 'Complejo de implementar', 'No separa test final'],
        useCase: 'Cuando hay pocos datos o se necesita validación robusta',
      },
    ],
    sections: [
      {
        title: 'Introducción al Dataset NSL-KDD',
        content:
          'El NSL-KDD es una versión mejorada del dataset KDD Cup 1999, utilizado ampliamente en investigación de detección de intrusiones. Contiene 125,973 registros de conexiones de red con 42 características que incluyen: tipo de protocolo (tcp, udp, icmp), duración de conexión, bytes enviados/recibidos, flags, tasas de error, y estadísticas de conexiones al mismo host. El dataset está diseñado para entrenar sistemas IDS (Intrusion Detection Systems) capaces de distinguir entre tráfico normal y ataques maliciosos.',
      },
      {
        title: 'Importancia de la División Estratificada',
        content:
          'La división estratificada es crítica cuando trabajamos con datos desbalanceados o cuando ciertas características tienen distribuciones específicas que deben preservarse. En NSL-KDD, los tipos de protocolo (tcp: ~79%, udp: ~13%, icmp: ~8%) tienen proporciones importantes que deben mantenerse en train/validation/test. Si usamos división aleatoria simple, podríamos terminar con un conjunto de validación con 100% tcp, lo que produciría evaluaciones sesgadas del modelo.',
      },
      {
        title: 'Metodología de División 60/20/20',
        content:
          'Se implementa una división en dos etapas: (1) Separar 40% para test+validation usando stratify sobre protocol_type, (2) Dividir ese 40% en dos partes iguales (20% validation, 20% test) también estratificadas. Resultado final: Training set con 75,583 registros (60%), Validation set con 25,195 registros (20%), Test set con 25,195 registros (20%). Esta proporción es estándar en machine learning y permite suficientes datos para entrenamiento mientras reserva conjuntos independientes para validación y evaluación final.',
      },
      {
        title: 'Verificación y Validación de la División',
        content:
          'Se utiliza visualización con histogramas para verificar que las proporciones de protocol_type se mantienen en los tres conjuntos. Se calcula el error de estratificación comparando las distribuciones con chi-cuadrado. Un error < 1% confirma que la estratificación funcionó correctamente. También se verifica que no haya data leakage (registros duplicados entre conjuntos) y que la distribución de la variable objetivo (normal vs anomaly) también se mantenga balanceada.',
      },
    ],
    code_examples: [
      {
        title: 'Carga del Dataset NSL-KDD desde formato ARFF',
        code: `import arff
import pandas as pd

def load_kdd_dataset(data_path):
    """
    Carga el dataset NSL-KDD desde formato ARFF.
    
    Parameters:
    -----------
    data_path : str
        Ruta al archivo .arff
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con todas las características y etiquetas
    """
    with open(data_path, 'r') as train_set:
        datasets = arff.load(train_set)
    
    # Extraer nombres de atributos
    attributes = [attr[0] for attr in datasets["attributes"]]
    
    # Crear DataFrame
    df = pd.DataFrame(datasets["data"], columns=attributes)
    
    return df

# Cargar datos
nsl_kdd_df = load_kdd_dataset("datasets/NSL-KDD/KDDTrain+.arff")
print(f"Shape: {nsl_kdd_df.shape}")
print(f"Columnas: {nsl_kdd_df.columns.tolist()}")`,
        explanation:
          'La biblioteca liac-arff permite leer archivos ARFF (Attribute-Relation File Format) usados en Weka. Este formato es común en datasets de machine learning académicos. El código extrae los nombres de atributos y los datos, creando un DataFrame de pandas para análisis posterior.',
      },
      {
        title: 'Función de División Estratificada Reutilizable',
        code: `from sklearn.model_selection import train_test_split

def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    """
    Divide el dataset en train/validation/test con estratificación.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset completo
    rstate : int, default=42
        Semilla para reproducibilidad
    shuffle : bool, default=True
        Si mezclar datos antes de dividir
    stratify : str, optional
        Nombre de columna para estratificar
        
    Returns:
    --------
    tuple
        (train_set, val_set, test_set) con proporción 60/20/20
    """
    # Preparar columna de estratificación
    strat = df[stratify] if stratify else None
    
    # Primera división: 60% train, 40% temp (val+test)
    train_set, test_set = train_test_split(
        df, 
        test_size=0.4, 
        random_state=rstate,
        shuffle=shuffle,
        stratify=strat
    )
    
    # Segunda división: dividir temp en 50/50 (20% val, 20% test del total)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set,
        test_size=0.5, 
        random_state=rstate, 
        shuffle=shuffle,
        stratify=strat
    )
    
    return (train_set, val_set, test_set)

# Aplicar división estratificada
train_set, val_set, test_set = train_val_test_split(
    nsl_kdd_df,
    stratify="protocol_type"
)`,
        explanation:
          'Esta función encapsula la lógica de división en dos etapas, permitiendo reutilización en otros proyectos. El parámetro stratify acepta el nombre de cualquier columna, haciéndola flexible. El uso de random_state=42 asegura que múltiples ejecuciones produzcan los mismos conjuntos, crucial para reproducibilidad en investigación.',
      },
      {
        title: 'Análisis Estadístico de la División',
        code: `import numpy as np
from scipy.stats import chisquare

def analyze_split_quality(original_df, train_set, val_set, test_set, stratify_column):
    """
    Analiza la calidad de la división estratificada.
    """
    datasets = {
        'Original': original_df,
        'Train': train_set,
        'Validation': val_set,
        'Test': test_set
    }
    
    print("=" * 70)
    print("ANÁLISIS DE CALIDAD DE DIVISIÓN")
    print("=" * 70)
    
    # Tamaños de conjuntos
    print("\\nTAMAÑOS DE CONJUNTOS:")
    for name, data in datasets.items():
        pct = len(data) / len(original_df) * 100 if name != 'Original' else 100
        print(f"{name:12}: {len(data):6} registros ({pct:5.1f}%)")
    
    # Distribución de columna estratificada
    print(f"\\nDISTRIBUCIÓN DE '{stratify_column}':")
    print("-" * 70)
    
    original_dist = original_df[stratify_column].value_counts(normalize=True)
    
    for category in original_dist.index:
        print(f"\\n{category}:")
        for name, data in datasets.items():
            dist = data[stratify_column].value_counts(normalize=True)
            pct = dist.get(category, 0) * 100
            diff = pct - (original_dist[category] * 100)
            print(f"  {name:12}: {pct:5.2f}% (diff: {diff:+.2f}%)")
    
    # Test chi-cuadrado
    print("\\nTEST CHI-CUADRADO (vs. Original):")
    for name in ['Train', 'Validation', 'Test']:
        observed = datasets[name][stratify_column].value_counts()
        expected = original_dist * len(datasets[name])
        chi_stat, p_value = chisquare(observed, expected)
        print(f"  {name:12}: χ² = {chi_stat:.4f}, p-value = {p_value:.4f}")

# Ejecutar análisis
analyze_split_quality(nsl_kdd_df, train_set, val_set, test_set, 'protocol_type')`,
        explanation:
          'Esta función proporciona un análisis exhaustivo de la división, calculando proporciones exactas, diferencias porcentuales respecto al original, y realizando tests chi-cuadrado para validar estadísticamente que las distribuciones son similares. Un p-value > 0.05 indica que no hay diferencia significativa, confirmando una estratificación exitosa.',
      },
      {
        title: 'Visualización Comparativa de Distribuciones',
        code: `import matplotlib.pyplot as plt
import seaborn as sns

def plot_stratification_comparison(original_df, train_set, val_set, test_set, stratify_column):
    """
    Genera visualizaciones para verificar estratificación.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Comparación de Distribución: {stratify_column}', 
                 fontsize=16, fontweight='bold')
    
    datasets = [
        ('Original Dataset', original_df),
        ('Training Set (60%)', train_set),
        ('Validation Set (20%)', val_set),
        ('Test Set (20%)', test_set)
    ]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for ax, (title, data) in zip(axes.flat, datasets):
        counts = data[stratify_column].value_counts()
        percentages = counts / counts.sum() * 100
        
        bars = ax.bar(range(len(counts)), percentages, color=colors)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45)
        ax.set_ylabel('Porcentaje (%)')
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Añadir etiquetas de porcentaje
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{pct:.1f}%',
                   ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

# Generar visualización
fig = plot_stratification_comparison(
    nsl_kdd_df, train_set, val_set, test_set, 'protocol_type'
)
plt.show()`,
        explanation:
          'La visualización permite verificar visualmente que las proporciones son consistentes entre todos los conjuntos. Las barras de igual altura entre gráficos indican estratificación exitosa. Se añaden etiquetas de porcentaje para comparación cuantitativa directa.',
      },
    ],
    key_points: [
      'División estratificada 60/20/20 mantiene proporciones de protocol_type en todos los conjuntos',
      'NSL-KDD: 125,973 registros, 42 features (34 numéricas, 8 categóricas) para detección de intrusiones',
      'Stratified sampling reduce sampling bias y asegura conjuntos representativos',
      'random_state=42 garantiza reproducibilidad - mismos conjuntos en múltiples ejecuciones',
      'Verificación estadística con chi-cuadrado confirma distribuciones similares (p-value > 0.05)',
      'División en dos etapas: primero 60/40, luego dividir el 40% en 20/20',
      'Función reutilizable train_val_test_split() con parámetros configurables',
      'Training set: 75,583 | Validation set: 25,195 | Test set: 25,195 registros',
    ],
  },
  notebook_08: {
    notebook_id: 'notebook_08',
    title: '08 - Preparación del Dataset',
    description: 'Limpieza, transformación y encoding para preparar datos de machine learning',
    dataset_stats: {
      totalRecords: 75583,
      totalFeatures: 41,
      numericFeatures: 34,
      categoricalFeatures: 7,
      missingValues: 10359,
      classDistribution: [
        { label: 'Valores completos', count: 65224, percentage: 86.3 },
        { label: 'Con valores faltantes', count: 10359, percentage: 13.7 },
      ],
    },
    technical_details: {
      algorithms: ['SimpleImputer', 'OneHotEncoder', 'StandardScaler', 'RobustScaler'],
      complexity: 'O(n × m) donde n=registros, m=features',
      prerequisites: ['scikit-learn', 'pandas', 'numpy'],
    },
    method_comparisons: [
      {
        method: 'Eliminar Filas con NaN',
        description: 'Remover todos los registros que contengan al menos un valor faltante',
        advantages: ['Simple y rápido', 'No introduce suposiciones', 'Datos 100% reales'],
        disadvantages: ['Pérdida de información', 'Reduce tamaño del dataset', 'Puede introducir bias si los missing values no son aleatorios'],
        useCase: 'Cuando hay pocos valores faltantes (<5%) o son Missing Completely At Random (MCAR)',
      },
      {
        method: 'Imputación con Mediana (Utilizado)',
        description: 'Reemplazar valores faltantes con la mediana de cada columna',
        advantages: ['Preserva todos los registros', 'Robusto a outliers', 'Mantiene distribución central'],
        disadvantages: ['Introduce valores artificiales', 'Reduce varianza', 'Puede distorsionar correlaciones'],
        useCase: 'Features numéricas con distribuciones asimétricas o presencia de outliers',
      },
      {
        method: 'OneHotEncoder vs OrdinalEncoder',
        description: 'OneHot crea columnas binarias; Ordinal asigna enteros secuenciales',
        advantages: ['OneHot: No asume orden entre categorías', 'Ordinal: Reduce dimensionalidad'],
        disadvantages: ['OneHot: Aumenta dimensiones (curse of dimensionality)', 'Ordinal: Crea relaciones ordinales falsas'],
        useCase: 'OneHot para variables nominales (protocol_type), Ordinal para ordinales (education_level)',
      },
      {
        method: 'StandardScaler vs RobustScaler',
        description: 'Standard usa media/std; Robust usa mediana/IQR',
        advantages: ['Standard: Datos distribuidos normalmente', 'Robust: Resistente a outliers'],
        disadvantages: ['Standard: Sensible a outliers', 'Robust: Puede no normalizar completamente'],
        useCase: 'Standard para distribuciones normales, Robust para datos con outliers',
      },
    ],
    sections: [
      {
        title: '1. Análisis Exploratorio de Valores Faltantes',
        content:
          'Antes de aplicar técnicas de imputación, es crucial entender el patrón de valores faltantes. En el training set de NSL-KDD (75,583 registros), encontramos 10,359 valores faltantes (13.7%) distribuidos principalmente en las columnas src_bytes (1,887 NaN) y dst_bytes (8,011 NaN). El análisis del patrón sugiere que los valores faltantes son MAR (Missing At Random) ya que correlacionan con el tipo de ataque, donde ciertos ataques DoS no generan bytes de respuesta.',
      },
      {
        title: '2. Estrategias de Manejo de Valores Faltantes',
        content:
          'Se exploran tres enfoques principales: (1) Eliminación de filas - reduce el dataset a 65,224 registros, perdiendo 13.7% de datos valiosos. (2) Eliminación de columnas - eliminaría src_bytes y dst_bytes que son features críticas para detectar ataques de denegación de servicio. (3) Imputación - método preferido que preserva todos los registros y features. Se elige SimpleImputer con strategy="median" porque es robusto a la presencia de outliers en las distribuciones de bytes.',
      },
      {
        title: '3. Encoding de Variables Categóricas',
        content:
          'El dataset contiene 7 features categóricas: protocol_type (3 valores: tcp, udp, icmp), service (70 valores: http, ftp, smtp, etc.), flag (11 valores: SF, S0, REJ, etc.), land (2 valores), logged_in (2 valores), is_host_login (2 valores), is_guest_login (2 valores). Para protocol_type, service y flag se aplica OneHotEncoding generando 84 nuevas columnas binarias. Las variables binarias se dejan como están. OrdinalEncoder se evita porque protocol_type="tcp"=0, "udp"=1, "icmp"=2 crearía una relación ordinal falsa donde el modelo podría interpretar que udp está "entre" tcp e icmp.',
      },
      {
        title: '4. Escalado y Normalización de Features Numéricas',
        content:
          'Las features numéricas tienen rangos muy diferentes: duration [0-58,329], src_bytes [0-1,379,963,888], count [0-511]. Sin escalado, algoritmos como SVM, KNN y redes neuronales darían mayor peso a src_bytes simplemente por su magnitud. Se implementa StandardScaler para estandarizar (media=0, std=1) y RobustScaler alternativo que usa mediana y rango intercuartil, siendo menos sensible a los outliers extremos presentes en src_bytes y dst_bytes. Crítico: el escalado solo se aplica a X (features), nunca a y (labels).',
      },
      {
        title: '5. Separación de Features y Labels',
        content:
          'Es fundamental separar el dataframe en X (features) e y (labels) antes del preprocesamiento: X = train_set.drop("class", axis=1) contiene las 41 features predictoras, y = train_set["class"] contiene las etiquetas objetivo ("normal" o "anomaly"). Las transformaciones de imputación, encoding y escalado solo se aplican a X. Aplicar escalado a y distorsionaría las etiquetas y haría imposible la clasificación. Esta separación también facilita el uso de pipelines de sklearn que esperan X e y como argumentos independientes.',
      },
    ],
    code_examples: [
      {
        title: 'Análisis Detallado de Valores Faltantes',
        code: `import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_missing_values(df):
    """
    Genera análisis completo de valores faltantes.
    """
    # Calcular valores faltantes por columna
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
        'Data_Type': df.dtypes
    })
    
    # Filtrar solo columnas con missing values
    missing_data = missing_data[missing_data['Missing_Count'] > 0]
    missing_data = missing_data.sort_values('Missing_Count', ascending=False)
    
    print("=" * 80)
    print("ANÁLISIS DE VALORES FALTANTES")
    print("=" * 80)
    print(f"\\nTotal de registros: {len(df)}")
    print(f"Total de features: {len(df.columns)}")
    print(f"Registros con algún NaN: {df.isnull().any(axis=1).sum()} "
          f"({df.isnull().any(axis=1).sum()/len(df)*100:.2f}%)")
    print(f"\\nColumnas con valores faltantes: {len(missing_data)}")
    print("\\n" + missing_data.to_string(index=False))
    
    # Visualización
    if not missing_data.empty:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        
        # Gráfico de barras
        missing_data.plot(x='Column', y='Missing_Count', kind='barh', 
                         ax=ax[0], color='#e74c3c', legend=False)
        ax[0].set_xlabel('Cantidad de Valores Faltantes')
        ax[0].set_title('Valores Faltantes por Columna')
        
        # Matriz de correlación de missing values
        msno_matrix = df[missing_data['Column']].isnull().astype(int)
        sns.heatmap(msno_matrix.corr(), annot=True, fmt='.2f', 
                   cmap='RdYlGn_r', ax=ax[1], cbar_kws={'label': 'Correlación'})
        ax[1].set_title('Correlación entre Valores Faltantes')
        
        plt.tight_layout()
        return fig
    
    return None

# Ejecutar análisis
fig = analyze_missing_values(train_set)
if fig:
    plt.show()`,
        explanation:
          'Esta función proporciona un análisis exhaustivo de valores faltantes, calculando cantidades y porcentajes por columna. La matriz de correlación de missing values revela si ciertos NaN tienden a aparecer juntos, sugiriendo patrones MNAR (Missing Not At Random) que requieren técnicas de imputación más sofisticadas.',
      },
      {
        title: 'Imputación con SimpleImputer y Validación',
        code: `from sklearn.impute import SimpleImputer
import pandas as pd

def impute_and_validate(X_train, strategy='median'):
    """
    Imputa valores faltantes y valida el resultado.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Features de entrenamiento con valores faltantes
    strategy : str
        Estrategia de imputación: 'mean', 'median', 'most_frequent'
        
    Returns:
    --------
    pd.DataFrame
        Datos imputados con estadísticas de validación
    """
    print(f"Aplicando imputación con estrategia: {strategy}")
    print("=" * 60)
    
    # Separar columnas numéricas (SimpleImputer no admite categóricas)
    X_num = X_train.select_dtypes(include=['number'])
    X_cat = X_train.select_dtypes(exclude=['number'])
    
    # Estadísticas pre-imputación
    missing_before = X_num.isnull().sum().sum()
    print(f"Valores faltantes ANTES: {missing_before}")
    print(f"Features numéricas: {len(X_num.columns)}")
    
    # Crear y ajustar imputador
    imputer = SimpleImputer(strategy=strategy)
    imputer.fit(X_num)
    
    # Transformar datos
    X_num_imputed = imputer.transform(X_num)
    X_num_imputed = pd.DataFrame(X_num_imputed, 
                                  columns=X_num.columns, 
                                  index=X_num.index)
    
    # Estadísticas post-imputación
    missing_after = X_num_imputed.isnull().sum().sum()
    print(f"Valores faltantes DESPUÉS: {missing_after}")
    
    # Mostrar valores utilizados para imputación
    print("\\nValores de imputación por columna:")
    imputed_values = pd.DataFrame({
        'Feature': X_num.columns[X_num.isnull().any()],
        f'{strategy.capitalize()}': imputer.statistics_[X_num.isnull().any()]
    })
    print(imputed_values.to_string(index=False))
    
    # Recombinar con columnas categóricas
    X_imputed = pd.concat([X_num_imputed, X_cat], axis=1)
    
    return X_imputed

# Aplicar imputación
X_train_imputed = impute_and_validate(X_train, strategy='median')

# Verificar resultado
assert X_train_imputed.isnull().sum().sum() == 0, "Error: Aún hay valores NaN"
print("\\n✓ Imputación exitosa: 0 valores faltantes restantes")`,
        explanation:
          'SimpleImputer calcula la estadística elegida (mediana, media o moda) durante fit() y la aplica en transform(). Es crucial usar fit() solo con datos de entrenamiento para evitar data leakage - la mediana del conjunto de prueba no debe influir en la imputación del entrenamiento. La función valida que todos los NaN fueron eliminados y muestra los valores exactos utilizados para cada columna.',
      },
      {
        title: 'OneHotEncoder con Manejo de Categorías Desconocidas',
        code: `from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def encode_categorical_features(X_train, X_test=None):
    """
    Aplica OneHotEncoding a features categóricas con manejo robusto.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Datos de entrenamiento
    X_test : pd.DataFrame, optional
        Datos de prueba/validación
        
    Returns:
    --------
    tuple
        (X_train_encoded, X_test_encoded, encoder, feature_names)
    """
    # Identificar columnas categóricas
    cat_columns = X_train.select_dtypes(include=['object']).columns.tolist()
    num_columns = X_train.select_dtypes(exclude=['object']).columns.tolist()
    
    print("ENCODING DE VARIABLES CATEGÓRICAS")
    print("=" * 70)
    print(f"Features categóricas: {len(cat_columns)}")
    print(f"Categorías únicas por feature:")
    
    for col in cat_columns:
        n_unique = X_train[col].nunique()
        print(f"  {col:20}: {n_unique:3} categorías")
    
    # Crear encoder con handle_unknown='ignore'
    # Esto previene errores si el conjunto de test tiene categorías nuevas
    encoder = OneHotEncoder(
        drop='first',  # Evita multicolinealidad
        sparse_output=False,  # Retorna array denso
        handle_unknown='ignore'  # Ignora categorías desconocidas
    )
    
    # Ajustar solo con datos de entrenamiento
    encoder.fit(X_train[cat_columns])
    
    # Transformar entrenamiento
    X_train_cat_encoded = encoder.transform(X_train[cat_columns])
    
    # Obtener nombres de features generadas
    feature_names = encoder.get_feature_names_out(cat_columns)
    print(f"\\nFeatures generadas: {len(feature_names)}")
    print(f"Total de columnas: {len(num_columns)} numéricas + {len(feature_names)} encoded")
    
    # Crear DataFrames
    X_train_cat_df = pd.DataFrame(
        X_train_cat_encoded,
        columns=feature_names,
        index=X_train.index
    )
    X_train_encoded = pd.concat([X_train[num_columns], X_train_cat_df], axis=1)
    
    # Transformar test si se proporciona
    X_test_encoded = None
    if X_test is not None:
        X_test_cat_encoded = encoder.transform(X_test[cat_columns])
        X_test_cat_df = pd.DataFrame(
            X_test_cat_encoded,
            columns=feature_names,
            index=X_test.index
        )
        X_test_encoded = pd.concat([X_test[num_columns], X_test_cat_df], axis=1)
        
        print(f"\\nDatos de test transformados: {X_test_encoded.shape}")
    
    print(f"\\n✓ Encoding completado exitosamente")
    
    return X_train_encoded, X_test_encoded, encoder, feature_names

# Aplicar encoding
X_train_enc, X_val_enc, encoder, feature_names = encode_categorical_features(
    X_train_imputed, X_val
)`,
        explanation:
          'OneHotEncoder con drop="first" crea una columna binaria para cada categoría excepto una, para evitar la multicolinealidad (dummy variable trap). El parámetro handle_unknown="ignore" es crítico: si el conjunto de test contiene una categoría no vista en entrenamiento (ej: un nuevo tipo de servicio), simplemente se codifica como [0,0,0...] en lugar de lanzar un error. Esto hace el pipeline más robusto en producción.',
      },
      {
        title: 'Comparación de Métodos de Escalado',
        code: `from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

def compare_scaling_methods(X_numeric, feature='src_bytes'):
    """
    Compara visualmente diferentes métodos de escalado.
    """
    scalers = {
        'Original': None,
        'StandardScaler': StandardScaler(),
        'RobustScaler': RobustScaler(),
        'MinMaxScaler': MinMaxScaler()
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Comparación de Métodos de Escalado - Feature: {feature}',
                 fontsize=16, fontweight='bold')
    
    for ax, (name, scaler) in zip(axes.flat, scalers.items()):
        if scaler is None:
            data = X_numeric[feature]
            title = f'{name}\\nRango: [{data.min():.0f}, {data.max():.0f}]'
        else:
            scaler.fit(X_numeric)
            data_scaled = scaler.transform(X_numeric)
            data = pd.DataFrame(data_scaled, columns=X_numeric.columns)[feature]
            title = f'{name}\\nRango: [{data.min():.2f}, {data.max():.2f}]'
        
        # Histograma
        ax.hist(data, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Valor')
        ax.set_ylabel('Frecuencia')
        ax.grid(True, alpha=0.3)
        
        # Estadísticas
        stats_text = f'Media: {data.mean():.2f}\\nStd: {data.std():.2f}\\nMediana: {data.median():.2f}'
        ax.text(0.98, 0.97, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

# Seleccionar solo features numéricas
X_numeric = X_train_encoded.select_dtypes(include=['number'])

# Comparar métodos
fig = compare_scaling_methods(X_numeric, feature='src_bytes')
plt.show()

# Análisis detallado
print("COMPARACIÓN DE ESCALADORES")
print("=" * 70)
print("StandardScaler:")
print("  - Transforma a media=0, std=1")
print("  - Fórmula: (x - mean) / std")
print("  - Sensible a outliers extremos")
print("  - Mejor para: Distribuciones normales")
print()
print("RobustScaler:")
print("  - Usa mediana y rango intercuartil (IQR)")
print("  - Fórmula: (x - median) / IQR")
print("  - Robusto a outliers")
print("  - Mejor para: Datos con outliers")
print()
print("MinMaxScaler:")
print("  - Escala al rango [0, 1]")
print("  - Fórmula: (x - min) / (max - min)")
print("  - Muy sensible a outliers")
print("  - Mejor para: Cuando se necesita rango específico")`,
        explanation:
          'La visualización comparativa muestra cómo cada método afecta la distribución. StandardScaler centra los datos alrededor de 0, útil para algoritmos que asumen media cero. RobustScaler es similar pero usa estadísticas resistentes a outliers. MinMaxScaler comprime al rango [0,1], útil para redes neuronales con activación sigmoid. La elección depende de la presencia de outliers y los requerimientos del algoritmo.',
      },
    ],
    key_points: [
      'Training set: 75,583 registros con 10,359 valores faltantes (13.7%) en src_bytes y dst_bytes',
      'SimpleImputer con mediana preferible por robustez a outliers en distribuciones asimétricas',
      'OneHotEncoder genera 84 columnas binarias desde 7 features categóricas (protocol_type, service, flag)',
      'drop="first" en OneHotEncoder previene multicolinealidad (dummy variable trap)',
      'handle_unknown="ignore" permite manejar categorías nuevas en producción sin errores',
      'RobustScaler recomendado sobre StandardScaler cuando hay outliers extremos en los datos',
      'Separación X/y crítica: transformaciones solo a features, nunca a labels',
      'Escalar features mejora convergencia en SVM, KNN, redes neuronales y gradient descent',
    ],
  },
  notebook_09: {
    notebook_id: 'notebook_09',
    title: '09 - Transformadores y Pipelines Personalizados',
    description: 'Automatización del preprocesamiento con componentes reutilizables',
    dataset_stats: {
      totalRecords: 75583,
      totalFeatures: 119,
      numericFeatures: 34,
      categoricalFeatures: 85,
      missingValues: 0,
      classDistribution: [
        { label: 'Features numéricas originales', count: 34, percentage: 28.6 },
        { label: 'Features encoded (OneHot)', count: 85, percentage: 71.4 },
      ],
    },
    technical_details: {
      algorithms: ['Custom Transformers', 'Pipeline', 'ColumnTransformer', 'BaseEstimator'],
      complexity: 'O(n × m) donde n=registros, m=transformaciones',
      prerequisites: ['scikit-learn', 'pandas', 'numpy', 'joblib para persistencia'],
    },
    method_comparisons: [
      {
        method: 'Transformación Manual',
        description: 'Aplicar cada transformación paso a paso manualmente',
        advantages: ['Control total sobre cada paso', 'Fácil de debuggear paso a paso', 'No requiere conocimiento de API'],
        disadvantages: ['Código repetitivo y propenso a errores', 'Difícil mantener consistencia entre train/val/test', 'Risk de data leakage si se ajustan transformadores con datos incorrectos'],
        useCase: 'Proyectos pequeños de exploración o prototipos rápidos',
      },
      {
        method: 'Pipeline de scikit-learn (Utilizado)',
        description: 'Encadenar transformadores en secuencia automática',
        advantages: ['Automatiza flujo completo', 'Previene data leakage', 'Fácil de serializar y deployar', 'Consistencia garantizada'],
        disadvantages: ['Requiere entender API de sklearn', 'Menos flexible para lógica compleja', 'Debugging puede ser más difícil'],
        useCase: 'Proyectos productivos que requieren reproducibilidad y deployment',
      },
      {
        method: 'ColumnTransformer vs Pipeline Simple',
        description: 'ColumnTransformer aplica transformadores diferentes a columnas específicas',
        advantages: ['Maneja features heterogéneas (numéricas + categóricas)', 'Procesamiento paralelo optimizado', 'Sintaxis declarativa clara'],
        disadvantages: ['Más complejo de configurar', 'Requiere especificar columnas explícitamente', 'Mayor overhead de memoria'],
        useCase: 'Datasets con tipos de features mixtos que requieren preprocesamiento diferente',
      },
    ],
    sections: [
      {
        title: 'Arquitectura de Transformadores Personalizados',
        content:
          'Los transformadores en sklearn siguen el patrón de diseño Adapter, permitiendo integrar cualquier transformación personalizada en pipelines estándar. Un transformador debe heredar de BaseEstimator (proporciona get_params/set_params para grid search) y TransformerMixin (proporciona fit_transform automático). Los métodos obligatorios son: fit(X, y=None) que aprende parámetros de los datos, y transform(X) que aplica la transformación. Opcionalmente, inverse_transform(X) permite revertir la transformación.',
      },
      {
        title: 'Pipeline: Automatización de Flujo de Preprocesamiento',
        content:
          'Un Pipeline es una secuencia de transformadores seguida opcionalmente de un estimador final. Sintaxis: Pipeline([(nombre1, transformador1), (nombre2, transformador2), ...]). Cuando se llama fit(), sklearn ejecuta fit_transform() en cada transformador secuencialmente, pasando la salida de uno como entrada del siguiente. El último paso puede ser un modelo (ej: RandomForestClassifier). Ventaja clave: previene data leakage porque fit() solo ve datos de entrenamiento, pero transform() puede aplicarse a cualquier conjunto.',
      },
      {
        title: 'ColumnTransformer: Procesamiento Heterogéneo',
        content:
          'ColumnTransformer permite aplicar diferentes transformadores a diferentes subconjuntos de columnas. Sintaxis: ColumnTransformer([(nombre, transformador, columnas)]). Caso de uso NSL-KDD: aplicar SimpleImputer→RobustScaler a las 34 features numéricas, y OneHotEncoder a las 7 features categóricas, todo en paralelo. El resultado final es la concatenación horizontal de todas las transformaciones. Parámetro remainder="passthrough" preserva columnas no especificadas; remainder="drop" las elimina.',
      },
      {
        title: 'Persistencia y Deployment de Pipelines',
        content:
          'Los pipelines completos pueden serializarse con joblib.dump(pipeline, "model.pkl") y cargarse con joblib.load("model.pkl"). Esto permite entrenar en desarrollo y deployar en producción sin reescribir código. El pipeline serializado incluye todos los transformadores con sus parámetros aprendidos (ej: las medianas del imputer, las categorías del encoder). En producción, solo se llama pipeline.predict(X_new) que automáticamente aplica todas las transformaciones antes de predecir.',
      },
      {
        title: 'Detección y Corrección de Bugs en Transformadores',
        content:
          'Análisis del código revela dos bugs críticos en CustomOneHotEncoding: (1) Typo: "fir()" debería ser "fit()". Python lanzará AttributeError al intentar ajustar el transformador. (2) Typo: "exlude" debería ser "exclude". select_dtypes lanzará TypeError. (3) Advertencia de deprecación: sparse=False está deprecated, usar sparse_output=False. (4) Método "inclede" en drop() no existe, debería ser "inplace". Estos errores ilustran la importancia de testing riguroso y type hints.',
      },
    ],
    code_examples: [
      {
        title: 'Transformador Base: DeleteNanRows',
        code: `from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class DeleteNanRows(BaseEstimator, TransformerMixin):
    """
    Transformador que elimina filas con valores NaN.
    
    Este es el transformador más simple posible, útil para demonstrar
    la estructura básica de un transformador personalizado de sklearn.
    
    Atributos:
    ----------
    n_removed_ : int
        Número de filas eliminadas durante transform()
    """
    
    def __init__(self):
        """Constructor vacío - no hay hiperparámetros."""
        pass
    
    def fit(self, X, y=None):
        """
        No hay nada que aprender, simplemente retorna self.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Datos de entrenamiento
        y : array-like, optional
            Labels (ignoradas)
            
        Returns:
        --------
        self : object
            Transformador fitted
        """
        return self
    
    def transform(self, X):
        """
        Elimina filas que contienen al menos un valor NaN.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Datos a transformar
            
        Returns:
        --------
        pd.DataFrame
            Datos sin filas con NaN
        """
        X_clean = X.dropna()
        self.n_removed_ = len(X) - len(X_clean)
        
        if self.n_removed_ > 0:
            print(f"DeleteNanRows: Eliminadas {self.n_removed_} filas "
                  f"({self.n_removed_/len(X)*100:.2f}%)")
        
        return X_clean

# Ejemplo de uso
deleter = DeleteNanRows()
X_clean = deleter.fit_transform(X_train)
print(f"Shape original: {X_train.shape}")
print(f"Shape limpio: {X_clean.shape}")`,
        explanation:
          'Este transformador demuestra la estructura mínima requerida: heredar de las clases base, implementar fit() que retorna self, e implementar transform() que retorna los datos transformados. fit() es trivial porque no hay parámetros que aprender. El atributo n_removed_ (con trailing underscore) sigue la convención sklearn de marcar atributos aprendidos durante fit.',
      },
      {
        title: 'Transformador con Estado: CustomScaler',
        code: `from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class CustomScaler(BaseEstimator, TransformerMixin):
    """
    Wrapper personalizado para RobustScaler con funcionalidad extendida.
    
    Parameters:
    -----------
    with_centering : bool, default=True
        Si centrar los datos antes de escalar
    with_scaling : bool, default=True
        Si escalar los datos al rango intercuartil
    quantile_range : tuple, default=(25.0, 75.0)
        Rango de cuantiles para calcular IQR
    """
    
    def __init__(self, with_centering=True, with_scaling=True, 
                 quantile_range=(25.0, 75.0)):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.scaler_ = None
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        """
        Aprende la mediana y el IQR de cada feature.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Datos de entrenamiento (solo numéricas)
        y : array-like, optional
            Labels (ignoradas)
            
        Returns:
        --------
        self : object
            Transformador fitted con scaler_ ajustado
        """
        # Crear scaler con parámetros especificados
        self.scaler_ = RobustScaler(
            with_centering=self.with_centering,
            with_scaling=self.with_scaling,
            quantile_range=self.quantile_range
        )
        
        # Ajustar a los datos
        self.scaler_.fit(X)
        
        # Guardar nombres de features si es DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        
        # Imprimir estadísticas aprendidas
        print(f"CustomScaler fitted:")
        print(f"  Mediana (center_): {self.scaler_.center_[:3]}...")
        print(f"  IQR (scale_): {self.scaler_.scale_[:3]}...")
        
        return self
    
    def transform(self, X):
        """
        Aplica el escalado robusto: (X - mediana) / IQR
        
        Parameters:
        -----------
        X : pd.DataFrame
            Datos a escalar
            
        Returns:
        --------
        pd.DataFrame
            Datos escalados con mismos nombres de columnas e índice
        """
        if self.scaler_ is None:
            raise ValueError("Transformador no fitted. Llama fit() primero.")
        
        # Transformar datos
        X_scaled = self.scaler_.transform(X)
        
        # Retornar como DataFrame si entrada era DataFrame
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(
                X_scaled,
                columns=self.feature_names_,
                index=X.index
            )
        
        return X_scaled
    
    def inverse_transform(self, X_scaled):
        """
        Revierte el escalado: X = (X_scaled * IQR) + mediana
        """
        return self.scaler_.inverse_transform(X_scaled)

# Uso
scaler = CustomScaler(quantile_range=(10.0, 90.0))  # IQR más robusto
X_scaled = scaler.fit_transform(X_train_numeric)
X_original = scaler.inverse_transform(X_scaled)  # Recuperar datos originales`,
        explanation:
          'Este transformador demuestra cómo envolver un scaler existente agregando funcionalidad personalizada. En fit() se crea el RobustScaler interno y se aprenden los parámetros (mediana, IQR). Los atributos aprendidos (scaler_, feature_names_) llevan trailing underscore. La función inverse_transform() permite recuperar datos originales, útil para interpretar resultados.',
      },
      {
        title: 'Pipeline Completo con ColumnTransformer',
        code: `from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Definir columnas por tipo
numeric_features = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count',
    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate'
]

categorical_features = [
    'protocol_type', 'service', 'flag'
]

# Pipeline para features numéricas
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

# Pipeline para features categóricas
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Combinar ambos pipelines con ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop',  # Eliminar columnas no especificadas
    n_jobs=-1  # Procesar en paralelo
)

# Pipeline completo: preprocesamiento + modelo
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ))
])

# Entrenar en una sola línea
print("Entrenando pipeline completo...")
full_pipeline.fit(X_train, y_train)

# Evaluar
train_score = full_pipeline.score(X_train, y_train)
val_score = full_pipeline.score(X_val, y_val)

print(f"\\nResultados:")
print(f"  Train Accuracy: {train_score:.4f}")
print(f"  Validation Accuracy: {val_score:.4f}")

# Hacer predicciones (preprocesamiento automático)
predictions = full_pipeline.predict(X_test)

# Inspeccionar componentes del pipeline
print(f"\\nComponentes del pipeline:")
for name, step in full_pipeline.named_steps.items():
    print(f"  {name}: {type(step).__name__}")

# Acceder a transformadores individuales
imputer = full_pipeline.named_steps['preprocessor'].transformers_[0][1].named_steps['imputer']
print(f"\\nMedianas aprendidas (primeras 5):")
print(imputer.statistics_[:5])`,
        explanation:
          'Este ejemplo muestra un pipeline production-ready completo. ColumnTransformer aplica numeric_transformer a 34 features numéricas y categorical_transformer a 3 features categóricas en paralelo (n_jobs=-1). El resultado se concatena y alimenta al RandomForestClassifier. Una sola llamada a fit() entrena todo el pipeline, previniendo data leakage porque cada transformador solo ve datos de entrenamiento. predict() automáticamente preprocesa datos nuevos.',
      },
      {
        title: 'Persistencia y Deployment del Pipeline',
        code: `import joblib
import pandas as pd
from datetime import datetime
import os
import sklearn # Import to get sklearn version

def save_pipeline(pipeline, filepath, metadata=None):
    """
    Guarda pipeline con metadata para deployment.
    
    Parameters:
    -----------
    pipeline : Pipeline
        Pipeline entrenado
    filepath : str
        Ruta donde guardar el archivo .pkl
    metadata : dict, optional
        Información adicional (versión, fecha, métricas)
    """
    # Crear diccionario con pipeline y metadata
    model_data = {
        'pipeline': pipeline,
        'metadata': metadata or {},
        'saved_at': datetime.now().isoformat(),
        'sklearn_version': sklearn.__version__
    }
    
    # Serializar con joblib (más eficiente que pickle para sklearn)
    joblib.dump(model_data, filepath, compress=3)
    
    print(f"Pipeline guardado en: {filepath}")
    print(f"Tamaño del archivo: {os.path.getsize(filepath) / 1024:.2f} KB")

def load_pipeline(filepath):
    """
    Carga pipeline desde archivo .pkl
    
    Returns:
    --------
    tuple
        (pipeline, metadata)
    """
    model_data = joblib.load(filepath)
    
    print(f"Pipeline cargado desde: {filepath}")
    print(f"Guardado en: {model_data['saved_at']}")
    print(f"Versión sklearn: {model_data['sklearn_version']}")
    
    return model_data['pipeline'], model_data['metadata']

# Guardar pipeline entrenado
metadata = {
    'model_type': 'RandomForest IDS',
    'train_accuracy': train_score,
    'val_accuracy': val_score,
    'n_features': len(numeric_features) + len(categorical_features),
    'dataset': 'NSL-KDD',
    'version': '1.0.0'
}

save_pipeline(full_pipeline, 'nsl_kdd_model.pkl', metadata)

# Simular deployment: cargar y predecir
pipeline_loaded, meta = load_pipeline('nsl_kdd_model.pkl')

# Nuevos datos (sin preprocesar)
new_data = pd.DataFrame({
    'duration': [0, 5],
    'protocol_type': ['tcp', 'udp'],
    'service': ['http', 'dns'],
    'flag': ['SF', 'S0'], # Added 'flag' to match categorical_features
    'wrong_fragment': [0,0],
    'urgent': [0,0],
    'hot': [0,0],
    'num_failed_logins': [0,0],
    'logged_in': [0,1],
    'num_compromised': [0,0],
    'root_shell': [0,0],
    'su_attempted': [0,0],
    'num_root': [0,0],
    'num_file_creations': [0,0],
    'num_shells': [0,0],
    'num_access_files': [0,0],
    'num_outbound_cmds': [0,0],
    'is_host_login': [0,0],
    'is_guest_login': [0,0],
    'count': [1, 1],
    'srv_count': [1, 1],
    'serror_rate': [0.0, 0.0],
    'srv_serror_rate': [0.0, 0.0],
    'rerror_rate': [0.0, 0.0],
    'srv_rerror_rate': [0.0, 0.0],
    'same_srv_rate': [1.0, 1.0],
    'diff_srv_rate': [0.0, 0.0],
    'srv_diff_host_rate': [0.0, 0.0],
    'dst_host_count': [255, 1],
    'dst_host_srv_count': [255, 1],
    'dst_host_same_srv_rate': [1.0, 1.0],
    'dst_host_diff_srv_rate': [0.0, 0.0],
    'dst_host_same_src_port_rate': [0.0, 1.0],
    'dst_host_srv_diff_host_rate': [0.0, 0.0],
    'src_bytes': [0, 100], # Example values
    'dst_bytes': [0, 50], # Example values
    'land': [0,0],
    'current_month_logins': [0,0], # Assuming this feature exists and was dropped
    'num_shells': [0,0], # Assuming this feature exists and was dropped
    'num_access_files': [0,0], # Assuming this feature exists and was dropped
    'num_outbound_cmds': [0,0], # Assuming this feature exists and was dropped
    'is_host_login': [0,0], # Assuming this feature exists and was dropped
    'is_guest_login': [0,0], # Assuming this feature exists and was dropped
    # Add all other numeric features with default/example values to avoid errors
    'urgent': [0,0],
    'hot': [0,0],
    'num_failed_logins': [0,0],
    'num_compromised': [0,0],
    'root_shell': [0,0],
    'su_attempted': [0,0],
    'num_root': [0,0],
    'num_file_creations': [0,0],
})


# Predecir directamente
predictions = pipeline_loaded.predict(new_data)
probabilities = pipeline_loaded.predict_proba(new_data)

print(f"\\nPredicciones: {predictions}")
print(f"Probabilidades:")
print(f"  Normal: {probabilities[:, 0]}")
print(f"  Anomaly: {probabilities[:, 1]}")`,
        explanation:
          'joblib es preferible a pickle para serializar modelos sklearn porque usa compresión y maneja arrays numpy eficientemente. El pipeline serializado contiene todos los transformadores con sus parámetros aprendidos, permitiendo aplicar exactamente las mismas transformaciones en producción. Metadata adicional ayuda con versionado y debugging. En deployment, solo se necesita cargar el .pkl y llamar predict() - todo el preprocesamiento ocurre automáticamente.',
      },
    ],
    key_points: [
      'Transformadores personalizados deben heredar de BaseEstimator y TransformerMixin',
      'fit() aprende parámetros de datos de entrenamiento, transform() los aplica a cualquier conjunto',
      'Atributos aprendidos llevan trailing underscore (ej: scaler_, feature_names_)',
      'Pipeline previene data leakage: fit() solo ve train, pero transform() funciona en train/val/test',
      'ColumnTransformer procesa features heterogéneas en paralelo (numéricas + categóricas)',
      'joblib.dump() serializa pipelines completos para deployment sin reescribir código',
      'BUGS detectados: fir()→fit(), exlude→exclude, sparse→sparse_output, inclede→inplace',
      'Pipeline completo: Imputer → Scaler → Encoder → Classifier en una sola línea de código',
      'remainder="passthrough" preserva columnas no especificadas en ColumnTransformer',
      'inverse_transform() permite recuperar datos originales desde transformados',
    ],
  },
}
