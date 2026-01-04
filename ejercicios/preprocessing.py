import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.encoder.fit(X)
        return self

    def transform(self, X):
        data = self.encoder.transform(X)
        cols = self.encoder.get_feature_names_out(X.columns)
        return pd.DataFrame(data, columns=cols, index=X.index)


class DataFramePreparer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        num_cols = X.select_dtypes(exclude="object").columns
        cat_cols = X.select_dtypes(include="object").columns

        self.pipeline = ColumnTransformer([
            ("num", num_pipeline, num_cols),
            ("cat", CustomOneHotEncoder(), cat_cols)
        ])

        self.pipeline.fit(X)
        self.columns = pd.get_dummies(X).columns
        return self

    def transform(self, X):
        data = self.pipeline.transform(X)
        return pd.DataFrame(data, columns=self.columns, index=X.index)
