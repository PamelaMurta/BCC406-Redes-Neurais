# 🎙️ Identificação de Falantes: Random Forest vs CNN

<div align="center">

**Comparação entre Random Forest e CNN para Identificação de Falantes usando VoxCeleb1**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-green.svg)](https://scikit-learn.org/)

</div>

---

## 📋 Navegação Rápida

<table>
<tr>
<td width="50%">

### 🚀 **Início Rápido**
- **Primeira Vez?** → [GUIA_RAPIDO.md](GUIA_RAPIDO.md)
- **Usar no Colab?** → [Tutorial Colab](notebooks/COLAB_Pipeline_Completo.md)
- **Executar Local?** → [Ver abaixo](#execução-local)

</td>
<td width="50%">

### 📚 **Documentação Completa**
- **Índice Geral** → [INDICE.md](INDICE.md)
- **Estrutura do Projeto** → [ESTRUTURA_DO_PROJETO.md](ESTRUTURA_DO_PROJETO.md)
- **Metodologia** → [docs/metodologia.md](docs/metodologia.md)

</td>
</tr>
</table>

---

## 🎯 Sobre o Projeto

Este projeto **compara duas abordagens** para identificação automática de falantes:

| Modelo | Tipo | Melhor Para |
|--------|------|-------------|
| **Random Forest** | Árvores de Decisão Ensemble | Datasets pequenos/médios, interpretabilidade |
| **CNN 1D** | Rede Neural Convolucional | Datasets grandes, padrões temporais complexos |

### 🔍 **O Que Foi Implementado**
✅ Extração de **40 MFCCs** (Mel-Frequency Cepstral Coefficients)  
✅ Pipeline completo de **treinamento e avaliação**  
✅ **Visualizações** detalhadas de métricas  
✅ Suporte para **VoxCeleb1** (38GB) ou **dados sintéticos** (180MB)  
✅ Execução **local** ou no **Google Colab**

---

## 📊 Resultados Esperados

### Dataset Sintético (200 amostras, 10 falantes)
```
🏆 Random Forest: 96.7% acurácia
📉 CNN 1D:        66.7% acurácia
Conclusão: RF vence em datasets pequenos
```

### VoxCeleb1 Completo (~1,200 falantes, ~100k arquivos)
*Resultados serão atualizados após treinamento*

---

## 🚀 Como Usar

### Opção 1️⃣: Google Colab (Recomendado - Sem Instalação)
```
1. Abra: notebooks/COLAB_Pipeline_Completo.md
2. Siga as instruções passo a passo
3. Execute no navegador (GPU gratuita!)
⏱️ Tempo estimado: 30-60 minutos
```

### Opção 2️⃣: Execução Local

#### **Instalação**
```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/BCC406-Redes-Neurais.git
cd BCC406-Redes-Neurais

# 2. Crie ambiente virtual
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Instale dependências
pip install -r requirements.txt
```

#### **Pipeline Completo** (Recomendado)
```bash
python scripts/run_full_pipeline.py
```

#### **Treinar Modelos Separadamente**
```bash
# Random Forest
python scripts/train_rf.py

# CNN
python scripts/train_cnn.py

# Avaliar ambos
python scripts/evaluate_models.py
```

### Opção 3️⃣: Notebooks Jupyter
```bash
jupyter notebook notebooks/
```
Execute na ordem: `01` → `02` → `03` → `04` → `05`

---

## 📁 Estrutura do Projeto

```
📦 BCC406-Redes-Neurais/
├── 📄 README.md                          # ← VOCÊ ESTÁ AQUI
├── 📄 INDICE.md                          # Índice completo da documentação
├── 📄 GUIA_RAPIDO.md                     # Guia de início rápido
├── 📄 ESTRUTURA_DO_PROJETO.md            # Descrição detalhada de cada arquivo
├── 📄 requirements.txt                   # Dependências Python
│
├── 📁 config/                            # Configurações
│   └── config.yaml                       # Hiperparâmetros e caminhos
│
├── 📁 data/                              # Datasets (não versionado - .gitignore)
│   ├── raw/                              # Áudio original
│   └── features/                         # Features extraídas (MFCCs)
│
├── 📁 docs/                              # Documentação técnica
│   └── metodologia.md                    # Metodologia detalhada
│
├── 📁 models/                            # Modelos treinados (não versionado)
│   ├── random_forest.pkl                 # Modelo RF salvo
│   └── cnn_modelo.h5                     # Modelo CNN salvo
│
├── 📁 notebooks/                         # Jupyter Notebooks
│   ├── 01_exploratory_analysis.ipynb     # Análise exploratória
│   ├── 02_feature_extraction.ipynb       # Extração de features
│   ├── 03_random_forest_baseline.ipynb   # Baseline RF
│   ├── 04_cnn_model.ipynb                # Modelo CNN
│   ├── 05_comparative_analysis.ipynb     # Análise comparativa
│   └── COLAB_Pipeline_Completo.md        # Tutorial para Colab
│
├── 📁 results/                           # Resultados (não versionado)
│   ├── figures/                          # Gráficos e visualizações
│   └── metrics/                          # Métricas de avaliação
│
├── 📁 scripts/                           # Scripts executáveis
│   ├── run_full_pipeline.py              # Pipeline completo
│   ├── train_rf.py                       # Treinar Random Forest
│   ├── train_cnn.py                      # Treinar CNN
│   ├── evaluate_models.py                # Avaliar modelos
│   ├── baixar_voxceleb1.py               # Baixar dataset VoxCeleb1
│   └── generate_synthetic_data.py        # Gerar dados sintéticos
│
└── 📁 src/                               # Código-fonte principal
    ├── data/                             # Gerenciamento de dados
    │   └── dataset.py                    # Classe SpeakerDataset
    ├── evaluation/                       # Avaliação
    │   ├── metrics.py                    # Cálculo de métricas
    │   └── visualization.py              # Gráficos
    ├── features/                         # Extração de features
    │   ├── audio_features.py             # MFCCs
    │   └── feature_aggregation.py        # Agregação
    ├── training/                         # Treinamento
    │   ├── trainer.py                    # Classe Trainer
    │   └── callbacks.py                  # TensorFlow callbacks
    └── utils/                            # Utilitários
        └── helpers.py                    # Funções auxiliares
```

---

## 🔧 Configuração

Edite `config/config.yaml` para ajustar:

```yaml
# Exemplo: Reduzir tempo de treinamento
training:
  rf:
    n_estimators: 50  # (padrão: 100)
  cnn:
    epochs: 20        # (padrão: 50)
```

---

## 📦 Dependências Principais

| Biblioteca | Versão | Uso |
|-----------|--------|-----|
| **TensorFlow** | 2.14.0 | Rede neural CNN |
| **scikit-learn** | 1.3.0 | Random Forest e métricas |
| **librosa** | 0.10.1 | Processamento de áudio |
| **NumPy** | 1.24.3 | Operações numéricas |
| **Matplotlib** | 3.7.2 | Visualizações |

**Total de dependências:** Ver [requirements.txt](requirements.txt)

---

## 🐛 Problemas Comuns

<details>
<summary><b>Erro: "No module named 'src'"</b></summary>

**Solução:**
```bash
# Execute a partir da raiz do projeto
cd BCC406-Redes-Neurais
python scripts/run_full_pipeline.py
```
</details>

<details>
<summary><b>Erro: "CUDA/GPU not found"</b></summary>

**Solução:** TensorFlow vai usar CPU automaticamente. Para GPU:
```bash
pip install tensorflow-gpu==2.14.0
```
Ou use o Google Colab (GPU gratuita).
</details>

<details>
<summary><b>Dataset VoxCeleb1 muito grande</b></summary>

**Solução:** Use dados sintéticos para testes:
```bash
python scripts/generate_synthetic_data.py
```
</details>

---

## 📖 Documentação Detalhada

| Documento | Descrição | Público |
|-----------|-----------|---------|
| [INDICE.md](INDICE.md) | Índice completo com navegação por nível/objetivo | Todos |
| [GUIA_RAPIDO.md](GUIA_RAPIDO.md) | Tutorial passo a passo (10 min) | Iniciantes |
| [ESTRUTURA_DO_PROJETO.md](ESTRUTURA_DO_PROJETO.md) | Descrição de cada arquivo e módulo | Desenvolvedores |
| [docs/metodologia.md](docs/metodologia.md) | Teoria e metodologia científica | Pesquisadores |
| [notebooks/COLAB_Pipeline_Completo.md](notebooks/COLAB_Pipeline_Completo.md) | Tutorial completo para Colab | Usuários Colab |

---

## 📝 Licença

Este projeto é parte do curso **BCC406 - Redes Neurais**.

---

## 🙏 Agradecimentos

- **Dataset:** [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) - University of Oxford
- **Bibliotecas:** TensorFlow, scikit-learn, librosa
- **Inspiração:** Artigos sobre Speaker Identification e Deep Learning

---

## 📞 Contato

**Dúvidas?** Abra uma issue ou consulte a [documentação completa](INDICE.md).

---

<div align="center">

**🎓 BCC406 - Redes Neurais**  
*Comparação de Modelos Clássicos vs Deep Learning*

</div>
