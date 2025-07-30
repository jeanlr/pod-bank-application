# Pod-Bank-Application
Planejamento e criação de um modelo que tenha capacidade de gerar um score de risco para contratação de empréstimo bancário.  

## Como replicar o projeto

1. **Crie e ative o ambiente virtual (.venv):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Abra um segundo terminal e execute o MLflow server:**
   ```bash
   mlflow server
   ```

> Recomenda-se manter o primeiro terminal com o ambiente `.venv` ativado para rodar os scripts do projeto, e o segundo terminal dedicado ao MLflow server para rastreamento dos experimentos referentes aos arquivos **train.py e predict.py**.

## CRISP-DM - Etapas do Projeto

### 1. Entendimento do Negócio
O objetivo é reduzir a inadimplência na concessão de crédito da Pod Bank, traduzindo a necessidade de negócio em um problema de dados: prever o risco de inadimplência de clientes a partir de suas características cadastrais e histórico financeiro.

### 2. Entendimento dos Dados
A análise exploratória foi realizada em [eda.ipynb](notebooks/eda.ipynb), avaliando o dataset `application_train.csv`. Foram identificados padrões, limitações (como variáveis com alto percentual de nulos) e oportunidades analíticas, além de entender a distribuição do target (inadimplência).

### 3. Preparação dos Dados
Os dados foram tratados e enriquecidos conforme [model-reg-log.ipynb](notebooks/model-reg-log.ipynb) e [train.py](notebooks/train.py):
- Remoção de variáveis temporárias e com alta correlação.
- Imputação de valores nulos (média para numéricas, moda para categóricas).
- Codificação de variáveis categóricas (Target Encoding, OneHot).
- Seleção de features relevantes via Random Forest e análise de importância.

### 4. Modelagem
Dois fluxos principais:
- Em [model-reg-log.ipynb](notebooks/model-reg-log.ipynb): Modelagem com Regressão Logística, categorização de variáveis via Decision Tree, ajuste e validação do scorecard.
- Em [train.py](notebooks/train.py): Modelagem com algoritmos de árvore (DecisionTree, RandomForest, GradientBoosting, LightGBM), otimização de hiperparâmetros com Optuna e validação cruzada.

### 5. Avaliação
A performance dos modelos foi avaliada por métricas técnicas (AUC-ROC, Gini, KS) e análise por decil, conforme relatórios e gráficos gerados nos notebooks. A escolha do modelo considerou tanto resultados quantitativos quanto impacto no negócio.

### 6. Deploy da Solução
O modelo final foi registrado e versionado via MLflow ([train.py](notebooks/train.py)), e o processo de escoragem foi implementado em [predict.py](notebooks/predict.py), permitindo a geração de scores para novos clientes e salvando os resultados em formato parquet para integração com sistemas do banco.

## 7. Apresentação executiva
Você pode verificar a apresentação [Projeto_Reducao_Inadimplencia_Pod_Bank.pptx](Projeto_Reducao_Inadimplencia_Pod_Bank.pptx), **onde estão os resultados das etapas**.

## Referências dos scripts principais
- [notebooks/eda.ipynb](notebooks/eda.ipynb): Análise exploratória dos dados.
- [notebooks/model-reg-log.ipynb](notebooks/model-reg-log.ipynb): Modelagem com Regressão Logística.
- [notebooks/train.py](notebooks/train.py): Modelagem com algoritmos de árvore e deploy via MLflow.
- [notebooks/predict.py](notebooks/predict.py): Pipeline de escoragem e geração de outputs.

---

## Portfólio e LinkedIn

- **Portfólio:** [https://jeanlr.github.io/portfolio-pessoal/](https://jeanlr.github.io/portifolio-pessoal/)
- **LinkedIn:** [https://www.linkedin.com/in/jeanlimarodovalho](https://www.linkedin.com/in/jeanlimarodovalho)
