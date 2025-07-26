# %%
from datetime import datetime
from warnings import filterwarnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TunedThresholdClassifierCV
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, f1_score, make_scorer, accuracy_score, precision_score, recall_score, auc
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from category_encoders import TargetEncoder
from category_encoders.woe import WOEEncoder
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
import sys
sys.path.append(r'/home/jean/projetos/pod-bank/global/')
from util import *
from lightgbm import LGBMClassifier
import pickle


import mlflow



filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# %%
mlflow.set_tracking_uri('http://localhost:5000')

client = mlflow.client.MlflowClient()
version = max([int(i.version) for i in client.get_latest_versions("application_model")])
version

# %%
loaded_model = mlflow.lightgbm.load_model(f"models:/application_model/{version}")
#%%
df_test = pd.read_parquet('../data/abt/abt_test', engine='fastparquet')

# %%
with open('../artifacts/prd_first_list_features.pkl', 'rb') as f:
    colunas_finais = pickle.load(f)
colunas_finais = [col for col in colunas_finais if col != 'TARGET']   
df_test = df_test[colunas_finais]
# %%
with open('../artifacts/prd_cols_to_keep_skl.pkl', 'rb') as f:
    cols_to_keep_train = pickle.load(f)
df_test = df_test.drop(columns=cols_to_keep_train)
df_test.shape
#%%
with open('../artifacts/prd_preprocesssor_skl.pkl', 'rb') as f:
    preprocesssor = pickle.load(f)
X_test_processed = preprocesssor.transform(df_test)
onehot_columns = []
if len(cat_features_low_card) > 0:
    onehot = preprocesssor.named_steps['preprocessor'].named_transformers_['cat_low'].named_steps['onehot']
    for i, col in enumerate(cat_features_low_card):
        cats = onehot.categories_[i]
        onehot_columns.extend([f"{col}_{cat}" for cat in cats])
processed_columns = onehot_columns + cat_features_high_card + num_features
X_test_processed = pd.DataFrame(X_test_processed, columns=processed_columns)
# %%
with open('../artifacts/prd_selected_features_skl.pkl', 'rb') as f:
    selected_features = pickle.load(f)
X_test_processed = X_test_processed[selected_features]
X_test_processed.shape

# %%
proba = loaded_model.predict_proba(X_test_processed)[:, 1]
proba

# %%
df_test['SCORE'] = proba
df_test['DATA_SCORE'] = datetime.today().date() 
df_test.head()

# %%
df_test.to_parquet(
    path='../data/escoragem/abt_test_com_score/',
    engine='fastparquet',                 
    index=False,
    partition_cols=['DATA_SCORE']
)