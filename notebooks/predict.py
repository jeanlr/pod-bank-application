# %% 
# Importação de bibliotecas e configurações iniciais
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

# Configurações iniciais para ignorar warnings e exibir todas as colunas do pandas
filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# %%
# Configuração do MLflow para rastreamento de experimentos
mlflow.set_tracking_uri('http://localhost:5000')

# Obtendo a versão mais recente do modelo registrado
client = mlflow.client.MlflowClient()
version = max([int(i.version) for i in client.get_latest_versions("application_model")])
version

# %%
# Carregando o modelo treinado mais recente do MLflow
loaded_model = mlflow.lightgbm.load_model(f"models:/application_model/{version}")

# %%
# Carregando os dados de teste para avaliação do modelo
df_test = pd.read_parquet('../data/abt/abt_test', engine='fastparquet')

# %%
# Carregando a lista de features finais usadas no modelo e preparando os dados de teste
with open('../artifacts/prd_first_list_features.pkl', 'rb') as f:
    colunas_finais = pickle.load(f)
    
# Removendo colunas de target e ID que não devem ser usadas como features
colunas_finais = [col for col in colunas_finais if col != 'TARGET']  
colunas_finais = [col for col in colunas_finais if col != 'SK_ID_CURR']  
df_test_02 = df_test[colunas_finais]

# %%
# Carregando e aplicando a seleção final de features usadas no treino
with open('../artifacts/prd_cols_to_keep_skl.pkl', 'rb') as f:
    cols_to_keep_train = pickle.load(f)
    
# Removendo colunas que não foram usadas no treino
df_test_02 = df_test_02.drop(columns=cols_to_keep_train)
df_test_02.shape
df_test_02

# %%
# Gerando metadados das features para identificar tipos e cardinalidades
metadados = generate_metadata(df=df_test_02, targets=[''], orderby='PC_NULOS')
metadados

# %%
# Separando features em categorias baseado nos metadados:
# - Categóricas com baixa cardinalidade (<20 categorias)
# - Categóricas com alta cardinalidade (>=20 categorias)
# - Numéricas
cat_features_low_card = metadados[(metadados['TIPO_FEATURE'] == 'object') & 
                                (metadados['CARDINALIDADE'] < 20)]['FEATURE'].tolist()

cat_features_high_card = metadados[(metadados['TIPO_FEATURE'] == 'object') & 
                                 (metadados['CARDINALIDADE'] >= 20)]['FEATURE'].tolist()

num_features = metadados[(metadados['TIPO_FEATURE'] != 'object')]['FEATURE'].tolist()

# %%
# Carregando e aplicando o pré-processador usado no treino
with open('../artifacts/prd_preprocesssor_skl.pkl', 'rb') as f:
    preprocesssor = pickle.load(f)
    
# Transformando os dados de teste
X_test_processed = preprocesssor.transform(df_test_02)

# Recuperando os nomes das colunas após one-hot encoding
onehot_columns = []
if len(cat_features_low_card) > 0:
    onehot = preprocesssor.named_steps['preprocessor'].named_transformers_['cat_low'].named_steps['onehot']
    for i, col in enumerate(cat_features_low_card):
        cats = onehot.categories_[i]
        onehot_columns.extend([f"{col}_{cat}" for cat in cats])
        
# Juntando todas as colunas processadas
processed_columns = onehot_columns + cat_features_high_card + num_features
X_test_processed = pd.DataFrame(X_test_processed, columns=processed_columns)

# %%
# Aplicando a seleção final de features usadas no modelo
with open('../artifacts/prd_selected_features_skl.pkl', 'rb') as f:
    selected_features = pickle.load(f)
X_test_processed = X_test_processed[selected_features]
X_test_processed.shape

# %%
# Obtendo as probabilidades de predição do modelo
proba = loaded_model.predict_proba(X_test_processed)[:, 1]
proba

# %%
# Adicionando os scores e data de processamento ao dataframe
df_test_02['SCORE'] = proba
df_test_02['DATA_SCORE'] = datetime.today().date() 
df_test_02 = df_test_02.sort_values(by='SCORE', ascending=False)
df_test_02.head()

# %%
# Adicionando o ID do cliente e selecionando apenas colunas relevantes para output
df_test_02 = df_test_02.join(df_test[['SK_ID_CURR']], how='left')
df_test_02 = df_test_02[['SK_ID_CURR', 'SCORE', 'DATA_SCORE']]
df_test_02.head()

# %%
# Salvando os resultados com particionamento pela data da escoragem
df_test_02.to_parquet(
    path='../data/escoragem/abt_test_com_score/',
    engine='fastparquet',                 
    index=False,
    partition_cols=['DATA_SCORE']
)