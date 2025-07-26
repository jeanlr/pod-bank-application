# %%
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
mlflow.set_experiment(experiment_id='364497209009674354')

#%%
df_treino_full = pd.read_parquet('../data/abt/abt_train', engine='fastparquet')


with open('../artifacts/prd_first_list_features.pkl', 'rb') as f:
    colunas_finais = pickle.load(f)
    
df_treino_full = df_treino_full[colunas_finais]


# Filtra as colunas que NÃO começam com "var_"
cols_to_keep_train = [col for col in df_treino_full.columns if col.startswith('var_')]

# Salva a remoção de variáveis na pasta artifacts
with open('../artifacts/prd_cols_to_keep_skl.pkl', 'wb') as f:
  pickle.dump(cols_to_keep_train, f)
  
len(cols_to_keep_train)

with open('../artifacts/prd_cols_to_keep_skl.pkl', 'rb') as f:
    cols_to_keep_train = pickle.load(f)
    
df_treino_full = df_treino_full.drop(columns=cols_to_keep_train)


# %%

# Separando as variáveis de entrada (features) e de saída (target)
X = df_treino_full.drop(columns=['TARGET','SK_ID_CURR'])
y = df_treino_full["TARGET"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42)
X_train.shape,X_test.shape

# %%

metadados = generate_metadata(df=X_train, targets=['TARGET'], orderby='PC_NULOS')
metadados

# %%

from sklearn.preprocessing import OneHotEncoder


cat_features_low_card = metadados[(metadados['TIPO_FEATURE'] == 'object') & 
                                (metadados['CARDINALIDADE'] < 20)]['FEATURE'].tolist()

cat_features_high_card = metadados[(metadados['TIPO_FEATURE'] == 'object') & 
                                 (metadados['CARDINALIDADE'] >= 20)]['FEATURE'].tolist()

num_features = metadados[(metadados['TIPO_FEATURE'] != 'object')]['FEATURE'].tolist()

# Definir pipelines separados para cada tipo de feature categórica
cat_pipe_low = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

cat_pipe_high = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('target_enc', TargetEncoder())
])

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

# Combinar todos os pipelines
preprocessor = ColumnTransformer([
    ('cat_low', cat_pipe_low, cat_features_low_card),
    ('cat_high', cat_pipe_high, cat_features_high_card),
    ('num', num_pipe, num_features)
])

preprocesssor = Pipeline(steps=[("preprocessor", preprocessor)])

# Aplicar o pré-processamento
X_train_processed = preprocesssor.fit_transform(X_train, y_train)
X_test_processed = preprocesssor.transform(X_test)

# Para obter os nomes das colunas após o OneHotEncoder
# (isso é mais complexo pois o OneHotEncoder cria múltiplas colunas)
onehot_columns = []
if len(cat_features_low_card) > 0:
    onehot = preprocesssor.named_steps['preprocessor'].named_transformers_['cat_low'].named_steps['onehot']
    for i, col in enumerate(cat_features_low_card):
        cats = onehot.categories_[i]
        onehot_columns.extend([f"{col}_{cat}" for cat in cats])

# Nomes finais das colunas
processed_columns = onehot_columns + cat_features_high_card + num_features

# Converter para DataFrame
X_train_processed = pd.DataFrame(X_train_processed, columns=processed_columns)
X_test_processed = pd.DataFrame(X_test_processed, columns=processed_columns)


with open('../artifacts/prd_processed_columns_skl.pkl', 'wb') as f:
  pickle.dump(processed_columns, f)
  
with open('../artifacts/prd_processed_columns_skl.pkl', 'rb') as f:
    processed_columns = pickle.load(f)
#%%
with open('../artifacts/prd_preprocesssor_skl.pkl', 'wb') as f:
  pickle.dump(preprocesssor, f)

# %%
clf = RandomForestClassifier(random_state=0, max_depth=5, min_samples_leaf=2)
clf.fit(X_train_processed, y_train)

#Obtendo o feature importance
feature_importances = clf.feature_importances_
features = pd.DataFrame({
    'Feature': X_train_processed.columns,
    'Importance': feature_importances
})

#Ordenar vars por importância
features = features.sort_values(by='Importance', ascending=False)

#Estabelecendo um ponto de corte
cutoff_maximp = 0.1

cutoff = cutoff_maximp * feature_importances.max()

# Selecionando vars acima do ponto de corte
selected_features = X_train_processed.columns[feature_importances > cutoff].tolist()
print('Número de features selecionadas: ', len(selected_features))

#Ordenar vars por importância
features = features.sort_values(by='Importance', ascending=True)

# Filtrar o DataFrame para apenas as features acima do corte
selected_features_df = features[features['Importance'] > cutoff]

# Ajusta o tamanho da figura com base no número de features selecionadas
plt.figure(figsize=(10, len(selected_features_df)*0.4))

# Plota as features selecionadas
plt.barh(selected_features_df['Feature'], selected_features_df['Importance'], color=(0.25, 0.5, 1))
plt.xlabel("Feature Importance")
plt.title("Variáveis Selecionadas - Random Forest")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %%
# Salva as features selecionadas na pasta artifacts
with open('../artifacts/prd_selected_features_skl.pkl', 'wb') as f:
  pickle.dump(selected_features, f)
  
# %%
algoritmos = [
    DecisionTreeClassifier(criterion='gini', random_state=0, max_depth=7, min_samples_leaf=3),
    RandomForestClassifier(random_state=0, max_depth=7, min_samples_leaf=3),
    GradientBoostingClassifier(random_state=0, max_depth=7, min_samples_leaf=3),
    LGBMClassifier(random_state=0, max_depth=7, min_child_samples=3, n_jobs=-1, verbosity=-1,)  # LightGBM
]

for algoritmo in algoritmos:

    nome_algoritmo = str(algoritmo)[:str(algoritmo).find("(")]
    # Treino do modelo
    algoritmo.fit(X_train_processed[selected_features],y_train)

    # Avaliar modelo
    metricas = calculate_metrics_models_classifier(nome_algoritmo,algoritmo, X_train_processed[selected_features], y_train, X_test_processed[selected_features], y_test)
    display(metricas)

# %%
import optuna

OPTUNA_EARLY_STOPING = 10

class EarlyStoppingExceeded(optuna.exceptions.OptunaError):
    early_stop = OPTUNA_EARLY_STOPING
    early_stop_count = 0
    best_score = None

def early_stopping_opt(study, trial):
    if EarlyStoppingExceeded.best_score == None:
      EarlyStoppingExceeded.best_score = study.best_value

    if study.best_value < EarlyStoppingExceeded.best_score:
        EarlyStoppingExceeded.best_score = study.best_value
        EarlyStoppingExceeded.early_stop_count = 0
    else:
      if EarlyStoppingExceeded.early_stop_count > EarlyStoppingExceeded.early_stop:
            EarlyStoppingExceeded.early_stop_count = 0
            best_score = None
            raise EarlyStoppingExceeded()
      else:
            EarlyStoppingExceeded.early_stop_count=EarlyStoppingExceeded.early_stop_count+1
    #print(f'EarlyStop counter: {EarlyStoppingExceeded.early_stop_count}, Best score: {study.best_value} and {EarlyStoppingExceeded.best_score}')
    return
  
  #%%
# Definir CV
from sklearn.model_selection import StratifiedKFold, cross_val_score


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Objetiva do Optuna
def objective(trial):
    params = {
        # Hiperparâmetros básicos
        'n_estimators': trial.suggest_int('n_estimators', 5, 50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        
        # Regularização
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        
        # Amostragem
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        
        # Balanceamento de classes
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 2, 10),
        
        # Tipo de boosting
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        
        # Fixos
        "metric": "auc",
        'objective': 'binary',
        'verbosity': -1,
        'random_state': 42
    }


    model = LGBMClassifier(**params)

    # cross_val_score retorna uma lista com a métrica para cada fold
    scores = cross_val_score(model, X_train_processed[selected_features], y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    # Média dos AUCs
    return np.mean(scores)

# Estudo
study = optuna.create_study(direction="maximize",study_name="modelo", storage="sqlite:///modelo.db")
study.add_trials(study.trials)
try:
    study.optimize(objective, n_trials=50, timeout=600, callbacks=[early_stopping_opt])

except EarlyStoppingExceeded:
    print(f'EarlyStopping Exceeded: No new best scores on iters {OPTUNA_EARLY_STOPING}')

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
    
# %%
algoritmo = LGBMClassifier(**study.best_params)

nome_algoritmo = str(algoritmo)[:str(algoritmo).find("(")]
        # Treino do modelo
algoritmo.fit(X_train_processed[selected_features],y_train)

        # Avaliar modelo
metricas = calculate_metrics_models_classifier(nome_algoritmo,algoritmo, X_train_processed[selected_features], y_train, X_test_processed[selected_features], y_test)
display(metricas)
    
# %%
import mlflow.sklearn

with mlflow.start_run():

    mlflow.sklearn.autolog()
    mlflow.lightgbm.autolog()
    
    algoritmo = LGBMClassifier(**study.best_params)

        # Treino do modelo
    algoritmo.fit(X_train_processed[selected_features],y_train)

    y_train_pred = algoritmo.predict(X_train_processed[selected_features])
    y_test_pred = algoritmo.predict(X_test_processed[selected_features])

        # Calculando as m�tricas para o conjunto de treino
    accuracy_train = accuracy_score(y_train, y_train_pred)
    precision_train = precision_score(y_train, y_train_pred)
    recall_train = recall_score(y_train, y_train_pred)
    auc_roc_train = roc_auc_score(y_train, algoritmo.predict_proba(X_train_processed[selected_features])[:, 1])

    # Calculando as m�tricas para o conjunto de teste
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred)
    recall_test = recall_score(y_test, y_test_pred)
    auc_roc_test = roc_auc_score(y_test, algoritmo.predict_proba(X_test_processed[selected_features])[:, 1])
    
    metrics = {
        "auc_roc_train": auc_roc_train,
        "auc_roc_test": auc_roc_test,
        "accuracy_train": accuracy_train,
        "accuracy_test": accuracy_test,
        "precision_train": precision_train,
        "precision_test": precision_test,
        "recall_train": recall_train,
        "recall_test": recall_test,
    }

    mlflow.log_metrics(metrics)

# %%
y_train.index = X_train_processed.index

bins = 16
tab = pd.concat([X_train_processed[selected_features],y_train],axis=1).copy()
tab['score'] = algoritmo.predict_proba(tab.drop(columns=['TARGET']))[:,0]
tab['decile'] = pd.qcut(tab['score'], bins, labels=False)

# Criar tabela detalhada
table = tab.groupby('decile').agg(
    min_score=pd.NamedAgg(column='score', aggfunc='min'),
    max_score=pd.NamedAgg(column='score', aggfunc='max'),
    event_rate=pd.NamedAgg(column='TARGET', aggfunc='mean'),
    volume=pd.NamedAgg(column='TARGET', aggfunc='size'),
    qt_bads=pd.NamedAgg(column='TARGET', aggfunc=lambda x: (x == 1).sum()),
    perc_total_bads=pd.NamedAgg(column='TARGET', aggfunc=lambda x: (x == 1).sum()/tab[tab.TARGET == 1].shape[0])
).reset_index()
table['min_score'] = 1000*table['min_score']
table['max_score'] = 1000*table['max_score']
table_train = table[['decile','event_rate','perc_total_bads', 'volume']]
table_train.rename(columns={'event_rate':'event_rate_train','perc_total_bads':'perc_total_bads_train', 'volume': 'volume_train'},inplace=True)
table_train

# %%
y_test.index = X_test_processed.index

bins = 16
tab = pd.concat([X_test_processed[selected_features],y_test],axis=1).copy()
tab['score'] = algoritmo.predict_proba(tab.drop(columns=['TARGET']))[:,0]
tab['decile'] = pd.qcut(tab['score'], bins, labels=False)

# Criar tabela detalhada
table = tab.groupby('decile').agg(
    min_score=pd.NamedAgg(column='score', aggfunc='min'),
    max_score=pd.NamedAgg(column='score', aggfunc='max'),
    event_rate=pd.NamedAgg(column='TARGET', aggfunc='mean'),
    volume=pd.NamedAgg(column='TARGET', aggfunc='size'),
    qt_bads=pd.NamedAgg(column='TARGET', aggfunc=lambda x: (x == 1).sum()),
    perc_total_bads=pd.NamedAgg(column='TARGET', aggfunc=lambda x: (x == 1).sum()/tab[tab.TARGET == 1].shape[0])
).reset_index()
table['min_score'] = 1000*table['min_score']
table['max_score'] = 1000*table['max_score']
table_test = table[['decile','event_rate','perc_total_bads', 'volume']]
table_test.rename(columns={'event_rate':'event_rate_test','perc_total_bads':'perc_total_bads_test', 'volume':'volume_test'},inplace=True)
table_test

# %%
summary = pd.merge(table_train,table_test,on='decile',how='inner')
summary

# %%# Plotando o gráfico de barras para Event Rate por Decil
barWidth = 0.3
r1 = np.arange(len(summary))
r2 = [x + barWidth for x in r1]

plt.bar(r1, summary['event_rate_train'], color='lightblue', width=barWidth, label='Train')
plt.bar(r2, summary['event_rate_test'], color='royalblue', width=barWidth, label='Test')

plt.xlabel('Decile')
plt.ylabel('Event Rate')
plt.title('Event Rate by Decile')
plt.xticks([r + barWidth for r in range(len(summary))], summary['decile'])
plt.legend()
plt.show()
