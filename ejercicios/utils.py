import pandas as pd
import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np


def load_kdd_dataset_from_arff(file_path):
    with open(file_path, 'r') as f:
        dataset = arff.load(f)
        attributes = [attr[0] for attr in dataset["attributes"]]
        return pd.DataFrame(dataset["data"], columns=attributes)


def split_dataset(df, stratify_col=None):
    strat = df[stratify_col] if stratify_col else None
    train, test = train_test_split(df, test_size=0.4, random_state=42, stratify=strat)
    strat = test[stratify_col] if stratify_col else None
    val, test = train_test_split(test, test_size=0.5, random_state=42, stratify=strat)
    return train, val, test

def cargar_archivo_subido(archivo):

    if archivo.name.endswith('.csv'):
        return pd.read_csv(archivo)

    elif archivo.name.endswith('.arff'):
        contenido = archivo.read().decode('utf-8')
        dataset = arff.loads(contenido)
        columnas = [a[0] for a in dataset['attributes']]
        return pd.DataFrame(dataset['data'], columns=columnas)

    return None
