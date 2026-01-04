from django.shortcuts import render
import arff
import pandas as pd
import io
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from .utils import cargar_archivo_subido, split_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from .preprocessing import DataFramePreparer
from io import TextIOWrapper
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tempfile
import os
from .spam_utils import create_prep_dataset



def home(request):
    return render(request, 'ejercicios/home.html')

def ejercicio5(request):
    contenido = None
    lineas = None

    if request.method == "POST" and request.FILES.get("dataset"):
        archivo = request.FILES["dataset"]

        # leer el archivo inmail.1
        contenido = archivo.read().decode("latin-1")

        # solo mostrar las primeras líneas para probar
        lineas = contenido.splitlines()[:20]

    return render(request, "ejercicios/ejercicio5.html", {
        "lineas": lineas
    })


def ejercicio6(request):
    tabla_html = None

    if request.method == 'POST' and request.FILES.get('dataset'):
        archivo = request.FILES['dataset']

        # CSV
        if archivo.name.endswith('.csv'):
            df = pd.read_csv(archivo)

        # ARFF (ESTA PARTE ES CLAVE)
        elif archivo.name.endswith('.arff'):
            contenido = archivo.read().decode('utf-8')
            data = arff.loads(contenido)
            columnas = [a[0] for a in data['attributes']]
            df = pd.DataFrame(data['data'], columns=columnas)

        # Convertir DataFrame a tabla HTML
        tabla_html = df.head(20).to_html(
            classes="table",
            border=0,
            index=False
        )

    return render(request, 'ejercicios/ejercicio6.html', {
        'tabla': tabla_html
    })



def ejercicio7(request):
    graficas = []

    if request.method == 'POST' and request.FILES.get('dataset'):
        # 1️⃣ Cargar dataset
        df = cargar_archivo_subido(request.FILES['dataset'])

        # 2️⃣ Particionar train, val, test con stratify
        train_set, val_set, test_set = split_dataset(df, stratify_col='protocol_type')

        # 3️⃣ Crear lista de conjuntos y títulos
        conjuntos = [
            (df, 'Dataset Completo'),
            (train_set, 'Train Set'),
            (val_set, 'Validation Set'),
            (test_set, 'Test Set')
        ]
        # 4️⃣ Generar las gráficas
        for data, titulo in conjuntos:
            plt.figure(figsize=(6,4))
            if 'protocol_type' in data.columns:
                data['protocol_type'].hist(color='skyblue', bins=len(data['protocol_type'].unique()))
            plt.title(titulo)
            plt.xlabel('Protocol Type')
            plt.ylabel('Frecuencia')
            plt.tight_layout()

            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()

            graficas.append({
                'titulo': titulo,
                'imagen': base64.b64encode(image_png).decode('utf-8')
            })

    return render(request, 'ejercicios/ejercicio7.html', {'graficas': graficas})




def ejercicio8(request):
    tabla_original = tabla_limpia = tabla_escalada = tabla_protocol = None
    longitudes = {}

    if request.method == 'POST' and request.FILES.get('dataset'):
        archivo = request.FILES['dataset']

        # Leer archivo
        if archivo.name.endswith('.csv'):
            df = pd.read_csv(archivo)
        elif archivo.name.endswith('.arff'):
            archivo_texto = io.TextIOWrapper(archivo.file, encoding='utf-8')
            data = arff.load(archivo_texto)
            columnas = [a[0] for a in data['attributes']]
            df = pd.DataFrame(data['data'], columns=columnas)

        # Particionar dataset (simple)
        n = len(df)
        train_set = df.iloc[:int(n*0.6)]
        val_set   = df.iloc[int(n*0.6):int(n*0.8)]
        test_set  = df.iloc[int(n*0.8):]
        longitudes = {"train": len(train_set), "val": len(val_set), "test": len(test_set)}

        # Limpiar datos: src_bytes y dst_bytes → numérico y rellenar NaN con media
        X_train = train_set.drop("class", axis=1, errors='ignore')
        for col in ["src_bytes", "dst_bytes"]:
            if col in X_train.columns:
                X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
                X_train[col] = X_train[col].fillna(X_train[col].mean())

        # Escalado robusto de src_bytes y dst_bytes
        if "src_bytes" in X_train.columns and "dst_bytes" in X_train.columns:
            scaler = RobustScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X_train[['src_bytes','dst_bytes']]),
                columns=['src_bytes','dst_bytes']
            )
        else:
            X_scaled = pd.DataFrame()

        # Codificación ordinal de protocol_type
        if 'protocol_type' in X_train.columns:
            ordinal_encoder = OrdinalEncoder()
            protocol_encode = pd.DataFrame(
                ordinal_encoder.fit_transform(X_train[['protocol_type']]),
                columns=['protocol_type_encoded']
            )
            tabla_protocol = pd.concat([X_train[['protocol_type']].head(10), protocol_encode.head(10)], axis=1).to_html(
                classes="table table-striped", index=False
            )

        # Preparar tablas para mostrar
        tabla_original = X_train.head(10).to_html(classes="table table-striped", index=False)
        tabla_limpia = X_train.head(10).to_html(classes="table table-striped", index=False)
        tabla_escalada = X_scaled.head(10).to_html(classes="table table-striped", index=False)

    return render(request, 'ejercicios/ejercicio8.html', {
        "longitudes": longitudes,
        "tabla_original": tabla_original,
        "tabla_limpia": tabla_limpia,
        "tabla_escalada": tabla_escalada,
        "tabla_protocol": tabla_protocol
    })

  


def ejercicio9(request):
    tabla_original = None
    tabla_transformada = None

    if request.method == 'POST' and request.FILES.get('dataset'):
        archivo = request.FILES['dataset']

        # --------- CARGA DATASET ---------
        if archivo.name.endswith('.csv'):
            df = pd.read_csv(archivo)

        elif archivo.name.endswith('.arff'):
            contenido = archivo.read().decode('utf-8', errors='ignore')
            data = arff.loads(contenido)

            columnas = [a[0] for a in data['attributes']]
            df = pd.DataFrame(data['data'], columns=columnas)

        # --------- SEPARAR X / y ---------
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # --------- TRAIN / TEST ---------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # --------- COLUMNAS ---------
        num_attribs = X_train.select_dtypes(include=['int64', 'float64']).columns
        cat_attribs = X_train.select_dtypes(include=['object']).columns

        # --------- PIPELINES ---------
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        full_pipeline = ColumnTransformer([
            ('num', num_pipeline, num_attribs),
            ('cat', cat_pipeline, cat_attribs)
        ])

        # --------- TRANSFORMACIÓN ---------
        X_train_prep = full_pipeline.fit_transform(X_train)

        X_train_prep = pd.DataFrame(
            X_train_prep.toarray() if hasattr(X_train_prep, "toarray") else X_train_prep,
            columns=list(pd.get_dummies(X_train).columns),
            index=X_train.index
        )

        # --------- TABLAS HTML ---------
        tabla_original = X_train.head(10).to_html(
            classes="table", border=0
        )

        tabla_transformada = X_train_prep.head(10).to_html(
            classes="table", border=0
        )

    return render(request, 'ejercicios/ejercicio9.html', {
        'tabla_original': tabla_original,
        'tabla_transformada': tabla_transformada
    })




def ejercicio10(request):
    tabla = None

    if request.method == "POST":
        archivo = request.FILES["dataset"]

        # 🔴 CONVERSIÓN CLAVE (bytes → texto)
        archivo_texto = TextIOWrapper(archivo.file, encoding="utf-8")

        data = arff.load(archivo_texto)
        columnas = [attr[0] for attr in data["attributes"]]
        df = pd.DataFrame(data["data"], columns=columnas)

        # Quitar clase
        X = df.drop("class", axis=1)

        # Preprocesamiento
        preparer = DataFramePreparer()
        X_prep = preparer.fit_transform(X)

        # Mostrar solo las primeras filas (como en la imagen)
        tabla = X_prep.head(10).to_html(
            classes="table",
            border=1
        )

    return render(request, "ejercicios/ejercicio10.html", {
        "tabla": tabla
    })