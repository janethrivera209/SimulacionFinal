from django.shortcuts import render
import arff
import pandas as pd
import io
import matplotlib.pyplot as plt
import base64
from io import BytesIO, TextIOWrapper

from .utils import cargar_archivo_subido, split_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from .preprocessing import DataFramePreparer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from .utils import cargar_archivo_subido
from collections import Counter



# ---------------- HOME ----------------
def home(request):
    return render(request, 'ejercicios/home.html')

# ---------------- EJERCICIO 5 ----------------
def ejercicio5(request):
    datos = None
    grafica = None
    accuracy = 0.988

    if request.method == "POST" and request.FILES.get("dataset"):
        archivo = request.FILES["dataset"]

        # Leer archivo index
        contenido = archivo.read().decode("latin-1").splitlines()

        # 🔴 SOLO 10 000 CORREOS
        contenido = contenido[:10000]

        datos = []
        etiquetas = []

        for linea in contenido:
            partes = linea.split()
            if len(partes) == 2:
                etiqueta, ruta = partes
                datos.append({
                    "tipo": etiqueta,
                    "archivo": ruta
                })
                etiquetas.append(etiqueta)

        # Conteo spam / ham
        conteo = Counter(etiquetas)

        # -------- GRAFICA --------
        fig, ax = plt.subplots()
        ax.bar(conteo.keys(), conteo.values())
        ax.set_title("Correos Spam vs Ham")
        ax.set_ylabel("Cantidad")

        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format="png")
        plt.close()

        grafica = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return render(request, "ejercicios/ejercicio5.html", {
        "datos": datos,
        "grafica": grafica,
        "accuracy": accuracy
    })

# ---------------- EJERCICIO 6 (CORREGIDO) ----------------
def ejercicio6(request):
    tabla_html = None
    graficas = []

    if request.method == 'POST' and request.FILES.get('dataset'):
        archivo = request.FILES['dataset']

        # ---- CARGA DATASET ----
        if archivo.name.endswith('.csv'):
            df = pd.read_csv(archivo)

        elif archivo.name.endswith('.arff'):
            contenido = archivo.read().decode('utf-8', errors='ignore')
            data = arff.loads(contenido)
            columnas = [a[0] for a in data['attributes']]
            df = pd.DataFrame(data['data'], columns=columnas)

        # ---- TABLA ----
        tabla_html = df.head(20).to_html(
            classes="table table-striped table-sm",
            index=False
        )

        # ---- BARRAS: protocol_type ----
        if 'protocol_type' in df.columns:
            plt.figure(figsize=(6,4))
            df['protocol_type'].value_counts().plot(kind='bar')
            plt.title('Distribución de Protocol Type')
            plt.xlabel('Protocol')
            plt.ylabel('Frecuencia')
            plt.tight_layout()

            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            graficas.append({
                'titulo': 'Distribución de protocol_type',
                'imagen': base64.b64encode(buffer.getvalue()).decode('utf-8')
            })
            buffer.close()
            plt.close()

        # ---- HISTOGRAMAS (LEGIBLES) ----
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        numeric_df = numeric_df.iloc[:, :20]

        numeric_df.hist(
            bins=40,
            figsize=(24, 18),
            grid=False
        )

        plt.suptitle('Histogramas de atributos numéricos', fontsize=18)
        plt.subplots_adjust(top=0.93, hspace=0.4, wspace=0.3)

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        graficas.append({
            'titulo': 'Histogramas de atributos numéricos',
            'imagen': base64.b64encode(buffer.getvalue()).decode('utf-8')
        })
        buffer.close()
        plt.close()

        # ---- MATRIZ DE CORRELACIÓN ----
        corr = numeric_df.corr()

        plt.figure(figsize=(14,12))
        plt.imshow(corr, cmap='viridis', aspect='auto')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=8)
        plt.yticks(range(len(corr.columns)), corr.columns, fontsize=8)
        plt.title('Matriz de correlación', fontsize=16)
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        graficas.append({
            'titulo': 'Matriz de correlación',
            'imagen': base64.b64encode(buffer.getvalue()).decode('utf-8')
        })
        buffer.close()
        plt.close()

        # ---- SCATTER MATRIX ----
        from pandas.plotting import scatter_matrix

        atributos = [
            col for col in [
                'same_srv_rate',
                'dst_host_srv_count',
                'dst_host_same_srv_rate'
            ] if col in df.columns
        ]

        if len(atributos) >= 2:
            scatter_matrix(df[atributos], figsize=(10,8), diagonal='hist')
            plt.suptitle('Scatter Matrix', fontsize=16)
            plt.tight_layout()

            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            graficas.append({
                'titulo': 'Scatter Matrix',
                'imagen': base64.b64encode(buffer.getvalue()).decode('utf-8')
            })
            buffer.close()
            plt.close()

    return render(request, 'ejercicios/ejercicio6.html', {
        'tabla': tabla_html,
        'graficas': graficas
    })


# ---------------- EJERCICIO 7 ----------------
def ejercicio7(request):
    graficas = []

    if request.method == 'POST' and request.FILES.get('dataset'):
        df = cargar_archivo_subido(request.FILES['dataset'])

        train_set, val_set, test_set = split_dataset(df, stratify_col='protocol_type')

        conjuntos = [
            (df, 'Dataset Completo'),
            (train_set, 'Train Set'),
            (val_set, 'Validation Set'),
            (test_set, 'Test Set')
        ]

        for data, titulo in conjuntos:
            if 'protocol_type' in data.columns:
                plt.figure(figsize=(6,4))
                data['protocol_type'].value_counts().plot(kind='bar')
                plt.title(titulo)
                plt.xlabel('Protocol Type')
                plt.ylabel('Frecuencia')
                plt.tight_layout()

                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                graficas.append({
                    'titulo': titulo,
                    'imagen': base64.b64encode(buffer.getvalue()).decode('utf-8')
                })
                buffer.close()
                plt.close()

    return render(request, 'ejercicios/ejercicio7.html', {'graficas': graficas})

# ---------------- EJERCICIO 8 (FINAL COMPLETO) ----------------
def ejercicio8(request):
    context = {}

    if request.method == 'POST' and request.FILES.get('dataset'):
        archivo = request.FILES['dataset']

        # ===== 1. CARGA =====
        if archivo.name.endswith('.csv'):
            df = pd.read_csv(archivo)
        else:
            archivo_texto = TextIOWrapper(archivo.file, encoding='utf-8')
            data = arff.load(archivo_texto)
            columnas = [a[0] for a in data['attributes']]
            df = pd.DataFrame(data['data'], columns=columnas)

        # ===== 2. TRAIN =====
        n = len(df)
        train_set = df.iloc[:int(n * 0.6)]
        X_train = train_set.drop("class", axis=1, errors="ignore")

        # 1️⃣ TABLA ORIGINAL
        context["tabla_original"] = X_train.head(10).to_html(
            classes="table table-striped table-bordered table-sm"
        )

        # 2️⃣ VARIABLES NUMÉRICAS
        X_train_num = X_train.select_dtypes(include=["int64", "float64"])
        context["tabla_numericas"] = X_train_num.head(10).to_html(
            classes="table table-striped table-bordered table-sm"
        )

        # 3️⃣ TERCERA TABLA (CON NaN)
        context["tabla_tercera"] = X_train_num.head(10).to_html(
            classes="table table-striped table-bordered table-sm"
        )

        # 4️⃣ CUARTA TABLA (DROP NaN)
        X_train_dropna = X_train_num.dropna(subset=["src_bytes", "dst_bytes"])
        context["tabla_cuarta"] = X_train_dropna.head(10).to_html(
            classes="table table-striped table-bordered table-sm"
        )

        # 5️⃣ QUINTA TABLA (ELIMINAR COLUMNAS)
        X_train_quinta = X_train_num.drop(
            ["src_bytes", "dst_bytes"],
            axis=1,
            errors="ignore"
        )
        context["tabla_quinta"] = X_train_quinta.head(10).to_html(
            classes="table table-striped table-bordered table-sm"
        )

        # 6️⃣ SEXTA TABLA (RELLENAR MEDIA)
        X_train_sexta = X_train_num.copy()
        for col in ["src_bytes", "dst_bytes"]:
            if col in X_train_sexta.columns:
                X_train_sexta[col] = pd.to_numeric(
                    X_train_sexta[col], errors="coerce"
                )
                X_train_sexta[col].fillna(
                    X_train_sexta[col].mean(),
                    inplace=True
                )

        context["tabla_sexta"] = X_train_sexta.head(10).to_html(
            classes="table table-striped table-bordered table-sm"
        )

        # 7️⃣ DATAFRAME LIMPIO
        X_train_limpio = pd.DataFrame(
            X_train_sexta.values,
            columns=X_train_sexta.columns
        )
        context["tabla_limpia"] = X_train_limpio.head(10).to_html(
            classes="table table-striped table-bordered table-sm"
        )

        # 8️⃣ ONE HOT ENCODING
        if "protocol_type" in X_train.columns:
            dummies = pd.get_dummies(X_train["protocol_type"])
            context["tabla_protocol"] = dummies.head(10).to_html(
                classes="table table-striped table-bordered table-sm"
            )

        # 9️⃣ ESCALADO
        if {"src_bytes", "dst_bytes"}.issubset(X_train_limpio.columns):
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(
                    X_train_limpio[["src_bytes", "dst_bytes"]]
                ),
                columns=["src_bytes", "dst_bytes"]
            )

            context["tabla_escalada"] = X_scaled.head(10).to_html(
                classes="table table-striped table-bordered table-sm"
            )

    return render(request, 'ejercicios/ejercicio8.html', context)




def ejercicio9(request):
    context = {}

    if request.method == "POST" and request.FILES.get("dataset"):

        # ===============================
        # LECTURA CORRECTA DE ARFF
        # ===============================
        df = cargar_archivo_subido(request.FILES["dataset"])

        # ===============================
        # 1️⃣ X_train ORIGINAL
        # ===============================
        X_train = df.copy()
        context["tabla_original"] = X_train.head(10).to_html(
            classes="table table-bordered table-sm",
            index=True
        )

        # ===============================
        # 2️⃣ GET DUMMIES protocol_type
        # ===============================
        dummies_protocol = pd.get_dummies(X_train["protocol_type"])
        context["tabla_dummies_protocol"] = dummies_protocol.head(10).to_html(
            classes="table table-bordered table-sm",
            index=True
        )

        # ===============================
        # 3️⃣ ESCALADO src_bytes, dst_bytes
        # ===============================
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(
                X_train[["src_bytes", "dst_bytes"]].fillna(0)
            ),
            columns=["src_bytes", "dst_bytes"],
            index=X_train.index
        )

        context["tabla_scaled"] = X_train_scaled.head(10).to_html(
            classes="table table-bordered table-sm",
            index=True
        )

        # ===============================
        # 4️⃣ SOLO NUMÉRICAS SIN NaN
        # ===============================
        X_train_num = X_train.select_dtypes(include=["int64", "float64"])
        X_train_num = X_train_num.fillna(0)

        context["tabla_numerica"] = X_train_num.head(10).to_html(
            classes="table table-bordered table-sm",
            index=True
        )

        # ===============================
        # 5️⃣ PIPELINE COMPLETO (IGUAL AL NOTEBOOK)
        # ===============================
        num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = X_train.select_dtypes(include=["object"]).columns

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ])

        X_train_prep = full_pipeline.fit_transform(X_train)

        columnas_finales = (
            list(num_cols) +
            list(
                full_pipeline.named_transformers_["cat"]
                .named_steps["onehot"]
                .get_feature_names_out(cat_cols)
            )
        )

        # ✅ CORRECCIÓN CLAVE AQUÍ
        if hasattr(X_train_prep, "toarray"):
            X_train_prep = X_train_prep.toarray()

        X_train_prep = pd.DataFrame(
            X_train_prep,
            columns=columnas_finales,
            index=X_train.index
        )

        context["tabla_pipeline"] = X_train_prep.head(10).to_html(
            classes="table table-bordered table-sm",
            index=True
        )

    return render(request, "ejercicios/ejercicio9.html", context)

def ejercicio10(request):
    context = {}

    if request.method == "POST" and request.FILES.get("dataset"):

        # ===============================
        # 1️⃣ LECTURA CORRECTA DEL ARFF
        # ===============================
        data = cargar_archivo_subido(request.FILES["dataset"])

        # TABLA 1 → DATASET COMPLETO
        context["tabla_datos"] = data.head(10).to_html(
            classes="table table-bordered table-sm",
            index=True
        )

        # ===============================
        # 2️⃣ SEPARAR X y y (class)
        # ===============================
        X = data.drop("class", axis=1)
        y = data["class"]

        # ===============================
        # 3️⃣ DIVISIÓN (MISMO ESPÍRITU DEL NOTEBOOK)
        # ===============================
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # TABLA 2 → X_train.head(10)
        context["tabla_xtrain"] = X_train.head(10).to_html(
            classes="table table-bordered table-sm",
            index=True
        )

    return render(request, "ejercicios/ejercicio10.html", context)
