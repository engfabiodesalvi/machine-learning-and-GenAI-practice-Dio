# Mentoria: Fundamentos de Dados para IA e Machine Learning

    A mentoria destaca que a qualidade dos dados é o fator mais crítico para o sucesso de modelos de Inteligência Artificial e Machine Learning. São abordadas as principais etapas de preparação de dados: análise exploratória, limpeza, tratamento de outliers, codificação de variáveis, normalização, balanceamento, seleção de atributos e validação. Demonstra-se na prática o uso dessas técnicas em um dataset de score de crédito. A mentora também orienta sobre competências essenciais para carreira em dados: Python, estatística, matemática, entendimento de negócio e aprendizado contínuo.

O mundo aprendeu que a tecnologia é um pilar essencial da sociedade e que as evoluções não vão parar. É necessário nunca parar de aprender e coaprender para que a transformação tecnológica aconteça. A DIO é uma startup brasileira focada em transformar vidas por meio da educação em tecnologia impulsionada por inteligência artificial, conectada a mais de 157 empresas globais, com milhões de bolsas distribuídas e impacto em milhões de pessoas.

A DIO já foi premiada como Tech Pioneers 2023 pelo World Economic Forum e atua globalmente conectando profissionais às empresas mais inovadoras do mundo. Sua missão é impulsionar a transformação tecnológica e o desenvolvimento de talentos.

Na sequência, inicia-se a mentoria sobre Fundamentos de Dados para Inteligência Artificial e Machine Learning. A convidada é Evelyn, arquiteta de soluções na AWS, com forte experiência em dados, analytics e machine learning. Ela destaca que a mentoria terá caráter introdutório, mas com base prática, incluindo demonstração em Jupyter Notebook.

Evelyn explica que muitos projetos focam excessivamente nos modelos e esquecem da qualidade dos dados. Dados mal tratados geram ruído, viés e inconsistência, e nenhum algoritmo sofisticado consegue compensar dados ruins. Modelos aprendem padrões a partir dos dados e podem amplificar erros existentes.

A preparação de dados começa com a análise exploratória, que permite entender variáveis, distribuições e possíveis problemas. Em seguida ocorre a limpeza dos dados, tratando valores inválidos, tipos incorretos e inconsistências. Também é necessário tratar valores ausentes (missing values), pois imputações inadequadas podem distorcer o modelo.

O tratamento de outliers é abordado por meio de técnicas como Z-score, intervalo interquartil (IQR), boxplot e winsorização. Transformações matemáticas e estatísticas, como uso de logaritmos, ajudam a reduzir efeitos de distribuições extremas.

Variáveis categóricas precisam ser transformadas em valores numéricos. Técnicas incluem one-hot encoding, label encoding e target encoding. A escolha depende da cardinalidade e do impacto no modelo. Label encoding pode induzir relação ordinal indevida entre categorias.

A normalização e padronização são utilizadas para manter as variáveis em escalas adequadas. Alguns métodos são sensíveis a outliers e exigem cuidado na escolha.

Na etapa de amostragem, os dados são divididos em conjuntos de treino e teste. O método holdout é comum, mas a validação cruzada (cross validation) melhora a estimativa do desempenho. Em problemas com classes desbalanceadas, a estratificação e técnicas como SMOTE ajudam a manter representatividade.

A seleção de atributos envolve análise de correlação, testes estatísticos (como qui-quadrado para variáveis categóricas) e análise de importância de features. Deve-se evitar colinearidade, que aumenta a variância do modelo. Métricas como VIF ajudam a detectar esse problema.

Outro cuidado é evitar data leakage, quando uma variável indevidamente antecipa o resultado. Técnicas como Lasso e SHAP auxiliam na avaliação da relevância das variáveis.

Evelyn também aborda técnicas de melhoria de modelos já treinados, como fine-tuning (especialização com novos dados) e destilação (transferência de conhecimento de um modelo maior para um menor).

Na demonstração prática, utiliza-se um dataset de classificação de score de crédito obtido do Kaggle. São realizadas etapas de análise exploratória, balanceamento de classes, tratamento de outliers, codificação de variáveis categóricas, seleção de atributos e divisão dos dados para treino e teste usando cross validation.

A mentoria finaliza com orientações de carreira. Para atuar na área de dados e machine learning, recomenda-se conhecimento em Python, matemática, estatística, entendimento de negócio e fundamentos de engenharia de dados. Transição de carreira exige dedicação e estudo contínuo.

---
### Anotações de Estudo
# Fundamentos de Dados para IA e Machine Learning

## 🎯 Objetivo
Garantir qualidade dos dados para maximizar desempenho, confiabilidade e generalização dos modelos de Machine Learning.

---

## ⚠️ Problemas de Dados Mal Tratados
- Ruído
- Viés
- Inconsistência
- Amplificação de erros pelo modelo
- Algoritmos avançados não compensam dados ruins

---

## 🔍 Etapas da Preparação de Dados

### 1. Análise Exploratória (EDA)
- Entender variáveis e distribuições
- Identificar valores extremos, nulos e padrões
- Visualizações: histogramas, boxplot, correlação

---

### 2. Limpeza de Dados
- Tipos incorretos (string em campo numérico)
- Valores inválidos (ex: negativos indevidos)
- Remoção ou correção de inconsistências

---

### 3. Tratamento de Missing Values
- Imputação (média, mediana, etc.)
- Remoção de linhas/colunas
- Avaliar impacto no modelo

---

### 4. Tratamento de Outliers
**Técnicas:**
- Z-score
- IQR (Q3 − Q1)
- Boxplot
- Winsorização
- Percentis

**Objetivo:**
- Evitar distorção do modelo
- Reduzir influência de extremos

---

### 5. Transformações Matemáticas
- Logaritmo
- Escalonamento
- Redução de assimetria
- Estabilização de variância

---

### 6. Codificação de Variáveis Categóricas
- One-Hot Encoding
- Label Encoding
- Target Encoding

**Cuidados:**
- Alta cardinalidade gera muitas colunas
- Label encoding pode gerar falsa ordem
- Avaliar impacto computacional

---

### 7. Normalização e Padronização
- Normalização (0 a 1)
- StandardScaler (média 0, desvio 1)
- RobustScaler (menos sensível a outliers)

---

### 8. Amostragem e Validação
- Holdout: 70–80% treino / 20–30% teste
- Cross Validation (K-Fold)
- Estratificação para classes desbalanceadas
- SMOTE para balanceamento

---

### 9. Avaliação de Erro
- Erro = Viés + Variância + Ruído
- Underfitting → alto viés
- Overfitting → alta variância
- Buscar equilíbrio

---

### 10. Seleção de Atributos (Feature Selection)
**Métodos:**
- Correlação
- Qui-quadrado
- Lasso
- SHAP
- Feature importance

**Cuidados:**
- Colinearidade
- VIF > 5 indica alerta
- Evitar data leakage
- Não usar apenas correlação linear

---

### 11. Melhoria de Modelos
- Fine-tuning → especialização com novos dados
- Destilação → transferência de conhecimento
- Fine-tuning reduz viés
- Destilação reduz variância

---

## 🧪 Demonstração Prática
- Dataset: Credit Score (Kaggle)
- Ferramentas:
  - Pandas
  - NumPy
  - Matplotlib
  - Scikit-learn
  - SageMaker
- Pipeline:
  1. EDA
  2. Limpeza
  3. Balanceamento
  4. Encoding
  5. Feature Selection
  6. Split e Cross Validation

---

## 👨‍💻 Competências para a Área
- Python
- Estatística (inferência, probabilidade)
- Matemática
- Engenharia de dados (Spark, pipelines)
- Entendimento de negócio
- Aprendizado contínuo
