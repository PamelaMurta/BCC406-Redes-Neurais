# 📂 Estrutura do Projeto - RF vs CNN para Identificação de Falantes

## 🎯 Visão Geral

Este documento descreve a organização completa do projeto.

---

## 📊 Estrutura de Diretórios

```
BCC406-Redes-Neurais/
│
├── 📁 config/                          # Configurações
│   └── config.yaml                     # Parâmetros centralizados do projeto
│
├── 📁 data/                            # Dados (não versionado)
│   ├── raw/                            # Dataset VoxCeleb1 bruto
│   └── processed/                      # Features extraídas
│
├── 📁 docs/                            # Documentação
│   └── metodologia.md                  # Metodologia detalhada do projeto
│
├── 📁 models/                          # Modelos treinados (não versionado)
│   ├── random_forest.pkl               # Modelo Random Forest
│   └── cnn_modelo.h5                   # Modelo CNN
│
├── 📁 notebooks/                       # Notebooks Jupyter
│   ├── 01_exploratory_analysis.ipynb   # Análise exploratória
│   ├── 02_feature_extraction.ipynb     # Extração de features
│   ├── 03_random_forest_baseline.ipynb # Treinamento RF
│   ├── 04_cnn_model.ipynb             # Treinamento CNN
│   ├── 05_comparative_analysis.ipynb   # Análise comparativa
│   └── COLAB_Pipeline_Completo.md      # 🌟 Pipeline completo Colab
│
├── 📁 results/                         # Resultados (não versionado)
│   └── comparison_results.pkl          # Comparação RF vs CNN
│
├── 📁 scripts/                         # Scripts executáveis
│   ├── baixar_voxceleb1.py            # 🌟 Download VoxCeleb1
│   ├── run_full_pipeline.py           # 🌟 Pipeline completo
│   ├── generate_synthetic_data.py      # Gerar dados sintéticos
│   ├── test_notebook01.py              # Testar instalação
│   ├── train_rf.py                     # Treinar Random Forest
│   ├── train_cnn.py                    # Treinar CNN
│   ├── evaluate_models.py              # Avaliar modelos
│   ├── run_01_extract_features.py      # Extrair features
│   └── download_data.sh                # Download (shell)
│
├── 📁 src/                             # Código fonte
│   ├── __init__.py
│   │
│   ├── data/                           # Módulo de dados
│   │   ├── __init__.py
│   │   └── dataset.py                  # Gerenciamento de dataset
│   │
│   ├── features/                       # Módulo de features
│   │   ├── __init__.py
│   │   ├── audio_features.py           # Extração de features de áudio
│   │   └── feature_aggregation.py      # Agregação para RF
│   │
│   ├── training/                       # Módulo de treinamento
│   │   ├── __init__.py
│   │   ├── trainer.py                  # Treinador genérico
│   │   └── callbacks.py                # Callbacks Keras
│   │
│   ├── evaluation/                     # Módulo de avaliação
│   │   ├── __init__.py
│   │   ├── metrics.py                  # Métricas de avaliação
│   │   └── visualization.py            # Visualizações
│   │
│   └── utils/                          # Utilitários
│       ├── __init__.py
│       └── helpers.py                  # Funções auxiliares
│
├── 📄 .gitignore                       # Arquivos ignorados pelo Git
├── 📄 GUIA_RAPIDO.md                   # 🌟 Guia de início rápido
├── 📄 IMPLEMENTATION_SUMMARY.md         # Resumo da implementação
├── 📄 README.md                        # 🌟 Documentação principal
└── 📄 requirements.txt                 # Dependências Python

```

**🌟 = Arquivos principais para começar**

---

## 🚀 Arquivos Principais

### 1. Para Começar Rapidamente

| Arquivo | Descrição | Quando Usar |
|---------|-----------|-------------|
| `GUIA_RAPIDO.md` | Instruções de execução rápida | Primeira vez no projeto |
| `README.md` | Documentação completa | Entender o projeto |
| `notebooks/COLAB_Pipeline_Completo.md` | Notebook completo para Colab | Executar no Colab |
| `scripts/run_full_pipeline.py` | Pipeline completo local | Executar tudo localmente |

### 2. Para Desenvolvimento

| Arquivo | Descrição | Quando Usar |
|---------|-----------|-------------|
| `config/config.yaml` | Configurações | Ajustar parâmetros |
| `src/data/dataset.py` | Dataset management | Trabalhar com dados |
| `src/features/audio_features.py` | Extração de features | Adicionar features |
| `scripts/generate_synthetic_data.py` | Dados de teste | Testar sem download |

### 3. Para Análise

| Arquivo | Descrição | Quando Usar |
|---------|-----------|-------------|
| `notebooks/01_exploratory_analysis.ipynb` | EDA | Explorar dados |
| `notebooks/05_comparative_analysis.ipynb` | Comparação | Analisar resultados |
| `src/evaluation/metrics.py` | Métricas | Avaliar modelos |

---

## 📦 Dados e Modelos

### Diretórios Não Versionados (excluídos do Git)

```
data/                   # Dataset e features (~40GB)
models/                 # Modelos treinados (~100MB)
results/                # Resultados e gráficos (~10MB)
venv/                   # Ambiente virtual
.mypy_cache/            # Cache do mypy
__pycache__/            # Cache Python
```

### Como Obter os Dados

```bash
# Opção 1: Download automático
python scripts/baixar_voxceleb1.py

# Opção 2: Dados sintéticos (para testes)
python scripts/generate_synthetic_data.py

# Opção 3: Upload manual para data/raw/
```

---

## 🔧 Fluxo de Trabalho

### 1. Primeira Execução

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Gerar dados de teste
python scripts/generate_synthetic_data.py

# 3. Executar pipeline completo
python scripts/run_full_pipeline.py
```

### 2. Desenvolvimento Iterativo

```bash
# 1. Explorar dados
jupyter notebook notebooks/01_exploratory_analysis.ipynb

# 2. Ajustar parâmetros
nano config/config.yaml

# 3. Treinar modelos específicos
python scripts/train_rf.py
python scripts/train_cnn.py

# 4. Avaliar
python scripts/evaluate_models.py
```

### 3. Produção com VoxCeleb1

```bash
# 1. Download dataset real
python scripts/baixar_voxceleb1.py

# 2. Pipeline completo
python scripts/run_full_pipeline.py

# 3. Analisar resultados
jupyter notebook notebooks/05_comparative_analysis.ipynb
```

---

## 📝 Documentação

### Documentos Principais

1. **README.md**
   - Visão geral do projeto
   - Instalação e configuração
   - Instruções de uso básico
   - Links importantes

2. **GUIA_RAPIDO.md**
   - Início rápido (3 opções)
   - Checklist de execução
   - FAQ
   - Solução de problemas

3. **IMPLEMENTATION_SUMMARY.md**
   - Resumo técnico da implementação
   - Estatísticas de código
   - Funcionalidades implementadas
   - Alinhamento com proposta

4. **docs/metodologia.md**
   - Metodologia detalhada
   - Fundamentação teórica
   - Detalhes de implementação
   - Referências bibliográficas

5. **notebooks/COLAB_Pipeline_Completo.md**
   - Tutorial completo para Colab
   - Código executável
   - Visualizações interativas
   - Resultados esperados

---

## 🧪 Scripts de Teste

| Script | Função | Tempo |
|--------|--------|-------|
| `test_notebook01.py` | Testar instalação | ~1 min |
| `generate_synthetic_data.py` | Gerar 200 amostras | ~5 seg |
| `run_full_pipeline.py` | Pipeline completo | ~10 min |

---

## 📊 Resultados Esperados

### Após Execução Completa

```
results/
├── comparison_results.pkl      # Comparação numérica
├── confusion_matrix_rf.png     # Matriz de confusão RF
├── confusion_matrix_cnn.png    # Matriz de confusão CNN
├── training_history.png        # Histórico de treinamento
└── comparative_analysis.html   # Relatório HTML

models/
├── random_forest.pkl           # Modelo RF treinado
├── cnn_modelo.h5               # Modelo CNN final
└── cnn_melhor.h5               # Melhor checkpoint CNN

data/processed/
├── X_train.npy                 # Features de treino
├── y_train.npy                 # Labels de treino
├── X_val.npy                   # Features de validação
├── y_val.npy                   # Labels de validação
├── X_test.npy                  # Features de teste
└── y_test.npy                  # Labels de teste
```

---

## 🔍 Navegação Rápida

### Por Tipo de Usuário

**Iniciante:**
1. Leia `README.md`
2. Siga `GUIA_RAPIDO.md`
3. Execute `notebooks/COLAB_Pipeline_Completo.md`

**Desenvolvedor:**
1. Clone o repositório
2. Configure ambiente (`requirements.txt`)
3. Explore `src/` e `scripts/`
4. Modifique `config/config.yaml`

**Pesquisador:**
1. Leia `docs/metodologia.md`
2. Analise `IMPLEMENTATION_SUMMARY.md`
3. Execute notebooks 01-05 sequencialmente
4. Revise resultados em `results/`

---

## 📞 Suporte

- **Documentação**: Comece pelo `README.md`
- **Problemas Comuns**: Veja `GUIA_RAPIDO.md` → FAQ
- **Issues**: GitHub Issues
- **Contato**: [Adicionar contato]

---

**Última atualização**: Dezembro 2025  
**Versão**: 1.0  
**Mantenedor**: Projeto BCC406
