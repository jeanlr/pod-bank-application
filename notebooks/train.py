# %% 
# Importação de bibliotecas e configurações iniciais
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

# Configurações para ignorar warnings e mostrar todas as colunas
filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# %%
# Configuração do MLflow para rastreamento de experimentos
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment(experiment_id='364497209009674354')

# %%
# Carregamento dos dados de treino e seleção das colunas finais
df_treino_full = pd.read_parquet('../data/abt/abt_train', engine='fastparquet')

# Carrega a lista de features finais do modelo
with open('../artifacts/prd_first_list_features.pkl', 'rb') as f:
    colunas_finais = pickle.load(f)
    
# Filtra apenas as colunas necessárias
df_treino_full = df_treino_full[colunas_finais]

# Filtra colunas que não começam com "var_" (removendo possíveis variáveis temporárias)
cols_to_keep_train = [col for col in df_treino_full.columns if col.startswith('var_')]

# Salva a lista de colunas a manter
with open('../artifacts/prd_cols_to_keep_skl.pkl', 'wb') as f:
  pickle.dump(cols_to_keep_train, f)
  
len(cols_to_keep_train)

# Carrega novamente para verificação
with open('../artifacts/prd_cols_to_keep_skl.pkl', 'rb') as f:
    cols_to_keep_train = pickle.load(f)
    
# Remove as colunas temporárias    
df_treino_full = df_treino_full.drop(columns=cols_to_keep_train)

# %%
# Preparação dos dados para modelagem
# Separação em features (X) e target (y)
X = df_treino_full.drop(columns=['TARGET','SK_ID_CURR'])
y = df_treino_full["TARGET"]

# Divisão em conjuntos de treino e teste
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42)
X_train.shape,X_test.shape

# %%
# Geração de metadados para análise das features
metadados = generate_metadata(df=X_train, targets=['TARGET'], orderby='PC_NULOS')
metadados

# %%
# Pré-processamento dos dados
from sklearn.preprocessing import OneHotEncoder

# Separação das features por tipo e cardinalidade
cat_features_low_card = metadados[(metadados['TIPO_FEATURE'] == 'object') & 
                                (metadados['CARDINALIDADE'] < 20)]['FEATURE'].tolist()

cat_features_high_card = metadados[(metadados['TIPO_FEATURE'] == 'object') & 
                                 (metadados['CARDINALIDADE'] >= 20)]['FEATURE'].tolist()

num_features = metadados[(metadados['TIPO_FEATURE'] != 'object')]['FEATURE'].tolist()

# Pipeline para features categóricas com baixa cardinalidade
cat_pipe_low = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Preenche valores faltantes com a moda
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # One-hot encoding
])

# Pipeline para features categóricas com alta cardinalidade
cat_pipe_high = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('target_enc', TargetEncoder())  # Target encoding para alta cardinalidade
])

# Pipeline para features numéricas
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Preenche valores faltantes com a mediana
    ('scaler', RobustScaler())  # Escalonamento robusto
])

# Combinação dos pipelines em um ColumnTransformer
preprocessor = ColumnTransformer([
    ('cat_low', cat_pipe_low, cat_features_low_card),
    ('cat_high', cat_pipe_high, cat_features_high_card),
    ('num', num_pipe, num_features)
])

# Pipeline final de pré-processamento
preprocesssor = Pipeline(steps=[("preprocessor", preprocessor)])

# Aplicação do pré-processamento
X_train_processed = preprocesssor.fit_transform(X_train, y_train)
X_test_processed = preprocesssor.transform(X_test)

# Recuperação dos nomes das colunas após one-hot encoding
onehot_columns = []
if len(cat_features_low_card) > 0:
    onehot = preprocesssor.named_steps['preprocessor'].named_transformers_['cat_low'].named_steps['onehot']
    for i, col in enumerate(cat_features_low_card):
        cats = onehot.categories_[i]
        onehot_columns.extend([f"{col}_{cat}" for cat in cats])

# Nomes finais das colunas
processed_columns = onehot_columns + cat_features_high_card + num_features

# Conversão para DataFrames
X_train_processed = pd.DataFrame(X_train_processed, columns=processed_columns)
X_test_processed = pd.DataFrame(X_test_processed, columns=processed_columns)

# Salvando as colunas processadas
with open('../artifacts/prd_processed_columns_skl.pkl', 'wb') as f:
  pickle.dump(processed_columns, f)
  
# Carregando para verificação
with open('../artifacts/prd_processed_columns_skl.pkl', 'rb') as f:
    processed_columns = pickle.load(f)

# %%
# Salvando o pré-processador
with open('../artifacts/prd_preprocesssor_skl.pkl', 'wb') as f:
  pickle.dump(preprocesssor, f)

# %%
# Seleção de features usando RandomForest
clf = RandomForestClassifier(random_state=0, max_depth=5, min_samples_leaf=2)
clf.fit(X_train_processed, y_train)

# Obtendo importância das features
feature_importances = clf.feature_importances_
features = pd.DataFrame({
    'Feature': X_train_processed.columns,
    'Importance': feature_importances
})

# Ordenando por importância
features = features.sort_values(by='Importance', ascending=False)

# Definindo ponto de corte para seleção (10% da importância máxima)
cutoff_maximp = 0.06
cutoff = cutoff_maximp * feature_importances.max()

# Selecionando features acima do corte
selected_features = X_train_processed.columns[feature_importances > cutoff].tolist()
print('Número de features selecionadas: ', len(selected_features))

# Visualização das features selecionadas
features = features.sort_values(by='Importance', ascending=True)
selected_features_df = features[features['Importance'] > cutoff]

plt.figure(figsize=(10, len(selected_features_df)*0.4))
plt.barh(selected_features_df['Feature'], selected_features_df['Importance'], color=(0.25, 0.5, 1))
plt.xlabel("Feature Importance")
plt.title("Variáveis Selecionadas - Random Forest")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %%
# Salvando as features selecionadas
with open('../artifacts/prd_selected_features_skl.pkl', 'wb') as f:
  pickle.dump(selected_features, f)
  
# %%
# Avaliação de diferentes algoritmos
algoritmos = [
    DecisionTreeClassifier(criterion='gini', random_state=0, max_depth=7, min_samples_leaf=3),
    RandomForestClassifier(random_state=0, max_depth=7, min_samples_leaf=3),
    GradientBoostingClassifier(random_state=0, max_depth=7, min_samples_leaf=3),
    LGBMClassifier(random_state=0, max_depth=7, min_child_samples=3, n_jobs=-1, verbosity=-1,)  # LightGBM
]

# Treino e avaliação de cada algoritmo
for algoritmo in algoritmos:
    nome_algoritmo = str(algoritmo)[:str(algoritmo).find("(")]
    # Treino do modelo
    algoritmo.fit(X_train_processed[selected_features], y_train)

    # Avaliação do modelo
    metricas = calculate_metrics_models_classifier(nome_algoritmo, algoritmo, 
                                                 X_train_processed[selected_features], y_train, 
                                                 X_test_processed[selected_features], y_test)
    display(metricas)

# %%
# Otimização de hiperparâmetros com Optuna
import optuna

OPTUNA_EARLY_STOPING = 10  # Número de iterações sem melhoria para parar

class EarlyStoppingExceeded(optuna.exceptions.OptunaError):
    early_stop = OPTUNA_EARLY_STOPING
    early_stop_count = 0
    best_score = None

def early_stopping_opt(study, trial):
    # Implementação do early stopping
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
    return
  
# %%
# Configuração do cross-validation
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Função objetivo para o Optuna
def objective(trial):
    # Espaço de busca de hiperparâmetros
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 5, 40),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 10, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'subsample': trial.suggest_float('subsample', 0.0, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 10),
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        "metric": "auc",
        'objective': 'binary',
        'verbosity': -1,
        'random_state': 42
    }

    model = LGBMClassifier(**params)

    # Avaliação com cross-validation
    scores = cross_val_score(model, X_train_processed[selected_features], y_train, 
                           cv=cv, scoring='roc_auc', n_jobs=-1)
    
    return np.mean(scores)  # Retorna a média do AUC

# Criação do estudo Optuna
study = optuna.create_study(direction="maximize", study_name="modelo", storage="sqlite:///modelo.db")
study.add_trials(study.trials)

try:
    # Execução da otimização
    study.optimize(objective, n_trials=50, timeout=800, callbacks=[early_stopping_opt])
except EarlyStoppingExceeded:
    print(f'EarlyStopping Exceeded: No new best scores on iters {OPTUNA_EARLY_STOPING}')

# Resultados da otimização
print("Number of finished trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
    
# %%
# Treino do modelo final com os melhores parâmetros
algoritmo = LGBMClassifier(**study.best_params)

nome_algoritmo = str(algoritmo)[:str(algoritmo).find("(")]
algoritmo.fit(X_train_processed[selected_features], y_train)

# Avaliação do modelo final
metricas = calculate_metrics_models_classifier(nome_algoritmo, algoritmo, 
                                             X_train_processed[selected_features], y_train, 
                                             X_test_processed[selected_features], y_test)
display(metricas)
    
# %%
# Registro do modelo no MLflow
import mlflow.sklearn

with mlflow.start_run():
    # Configuração do autolog
    mlflow.sklearn.autolog()
    mlflow.lightgbm.autolog()
    
    # Treino do modelo final
    algoritmo = LGBMClassifier(**study.best_params)
    algoritmo.fit(X_train_processed[selected_features], y_train)

    # Predições
    y_train_pred = algoritmo.predict(X_train_processed[selected_features])
    y_test_pred = algoritmo.predict(X_test_processed[selected_features])
    
    # Probabilidades preditas
    y_train_proba = algoritmo.predict_proba(X_train_processed[selected_features])[:, 1]
    y_test_proba = algoritmo.predict_proba(X_test_processed[selected_features])[:, 1]

    # Cálculo das métricas básicas
    accuracy_train = accuracy_score(y_train, y_train_pred)
    precision_train = precision_score(y_train, y_train_pred)
    recall_train = recall_score(y_train, y_train_pred)
    auc_roc_train = roc_auc_score(y_train, y_train_proba)

    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred)
    recall_test = recall_score(y_test, y_test_pred)
    auc_roc_test = roc_auc_score(y_test, y_test_proba)
    
    # Cálculo do GINI (GINI = 2*AUC - 1)
    gini_train = 2 * auc_roc_train - 1
    gini_test = 2 * auc_roc_test - 1
    
    # Cálculo do KS
    def calculate_ks(y_true, y_proba):
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        return max(tpr - fpr)
    
    ks_train = calculate_ks(y_train, y_train_proba)
    ks_test = calculate_ks(y_test, y_test_proba)
    
    # Log das métricas no MLflow
    metrics = {
        "auc_roc_train": auc_roc_train,
        "auc_roc_test": auc_roc_test,
        "gini_train": gini_train,
        "gini_test": gini_test,
        "ks_train": ks_train,
        "ks_test": ks_test,
        "accuracy_train": accuracy_train,
        "accuracy_test": accuracy_test,
        "precision_train": precision_train,
        "precision_test": precision_test,
        "recall_train": recall_train,
        "recall_test": recall_test,
    }

    mlflow.log_metrics(metrics)

# %%
# Análise de performance por decil - conjunto de treino
y_train.index = X_train_processed.index

bins = 10  # Número de decis
tab = pd.concat([X_train_processed[selected_features], y_train], axis=1).copy()
tab['score'] = algoritmo.predict_proba(tab.drop(columns=['TARGET']))[:,0]  # Probabilidade de não evento
tab['decile'] = pd.qcut(tab['score'], bins, labels=False)  # Divisão em decis

# Cálculo de métricas por decil
table = tab.groupby('decile').agg(
    min_score=pd.NamedAgg(column='score', aggfunc='min'),
    max_score=pd.NamedAgg(column='score', aggfunc='max'),
    event_rate=pd.NamedAgg(column='TARGET', aggfunc='mean'),  # Taxa de eventos (bad rate)
    volume=pd.NamedAgg(column='TARGET', aggfunc='size'),  # Número de observações
    qt_bads=pd.NamedAgg(column='TARGET', aggfunc=lambda x: (x == 1).sum()),  # Número de eventos
    perc_total_bads=pd.NamedAgg(column='TARGET', aggfunc=lambda x: (x == 1).sum()/tab[tab.TARGET == 1].shape[0])  # % do total de eventos
).reset_index()

# Ajustes de formatação
table['min_score'] = 1000*table['min_score']
table['max_score'] = 1000*table['max_score']
table_train = table[['decile','event_rate','perc_total_bads', 'volume']]
table_train.rename(columns={'event_rate':'event_rate_train','perc_total_bads':'perc_total_bads_train', 'volume': 'volume_train'}, inplace=True)
table_train

# %%
# Análise de performance por decil - conjunto de teste
y_test.index = X_test_processed.index

bins = 10
tab = pd.concat([X_test_processed[selected_features], y_test], axis=1).copy()
tab['score'] = algoritmo.predict_proba(tab.drop(columns=['TARGET']))[:,0]
tab['decile'] = pd.qcut(tab['score'], bins, labels=False)

# Cálculo de métricas por decil
table = tab.groupby('decile').agg(
    min_score=pd.NamedAgg(column='score', aggfunc='min'),
    max_score=pd.NamedAgg(column='score', aggfunc='max'),
    event_rate=pd.NamedAgg(column='TARGET', aggfunc='mean'),
    volume=pd.NamedAgg(column='TARGET', aggfunc='size'),
    qt_bads=pd.NamedAgg(column='TARGET', aggfunc=lambda x: (x == 1).sum()),
    perc_total_bads=pd.NamedAgg(column='TARGET', aggfunc=lambda x: (x == 1).sum()/tab[tab.TARGET == 1].shape[0])
).reset_index()

# Ajustes de formatação
table['min_score'] = 1000*table['min_score']
table['max_score'] = 1000*table['max_score']
table_test = table[['decile','event_rate','perc_total_bads', 'volume']]
table_test.rename(columns={'event_rate':'event_rate_test','perc_total_bads':'perc_total_bads_test', 'volume':'volume_test'}, inplace=True)
table_test

# %%
# Consolidação dos resultados de treino e teste
summary = pd.merge(table_train, table_test, on='decile', how='inner')
summary

# %%
# Visualização comparativa do event rate entre treino e teste
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