# Comparação entre Random Forest e Redes Neurais Convolucionais para Identificação de Falantes

## Sobre o Projeto

Este projeto implementa um estudo comparativo entre modelos clássicos de Machine Learning (Random Forest) e Deep Learning (CNN 1D) para identificação de falantes em condições não controladas, utilizando o dataset VoxCeleb1.

### Objetivo

Avaliar e comparar o desempenho de dois paradigmas de aprendizado de máquina na tarefa de identificação de falantes com um regime de poucos falantes (5-10 falantes):
- **Baseline Clássico**: Random Forest com features acústicas agregadas
- **Deep Learning**: Rede Neural Convolucional 1D com features sequenciais

## Estrutura do Projeto

```
.
├── README.md                           # Este arquivo
├── requirements.txt                    # Dependências Python
├── config/
│   └── config.yaml                     # Configurações centralizadas
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb   # Análise exploratória dos dados
│   ├── 02_feature_extraction.ipynb     # Extração de features
│   ├── 03_random_forest_baseline.ipynb # Treinamento Random Forest
│   ├── 04_cnn_model.ipynb             # Treinamento CNN
│   └── 05_comparative_analysis.ipynb   # Análise comparativa
├── src/
│   ├── data/                          # Módulo de processamento de dados
│   │   ├── preprocessing.py           # Pré-processamento de áudio
│   │   ├── download_voxceleb.py       # Download do dataset
│   │   └── dataset.py                 # Gerenciamento de datasets
│   ├── features/                      # Módulo de extração de features
│   │   ├── audio_features.py          # Extração de features acústicas
│   │   └── feature_aggregation.py     # Agregação para RF
│   ├── models/                        # Módulo de modelos
│   │   ├── base_model.py              # Classe base
│   │   ├── random_forest.py           # Modelo Random Forest
│   │   └── cnn_1d.py                  # Modelo CNN 1D
│   ├── training/                      # Módulo de treinamento
│   │   ├── trainer.py                 # Treinador genérico
│   │   └── callbacks.py               # Callbacks personalizados
│   ├── evaluation/                    # Módulo de avaliação
│   │   ├── metrics.py                 # Cálculo de métricas
│   │   └── visualization.py           # Visualizações
│   └── utils/
│       └── helpers.py                 # Funções auxiliares
├── scripts/
│   ├── download_data.sh               # Script para download dos dados
│   ├── train_rf.py                    # Treinar Random Forest
│   ├── train_cnn.py                   # Treinar CNN
│   └── evaluate_models.py             # Avaliar modelos
└── docs/
    └── metodologia.md                 # Documentação detalhada
```

## Instalação

### Requisitos

- Python 3.8 ou superior
- pip ou conda
- FFmpeg (para conversão de áudio)

### Configuração do Ambiente

1. Clone o repositório:
```bash
git clone https://github.com/PamelaMurta/BCC406-Redes-Neurais.git
cd BCC406-Redes-Neurais
```

2. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Instale o FFmpeg (se ainda não estiver instalado):
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows: Baixe de https://ffmpeg.org/download.html
```

## Preparação dos Dados

### Download do VoxCeleb1

1. Execute o script de download:
```bash
bash scripts/download_data.sh
```

2. O script irá:
   - Baixar o VoxCeleb1 development set
   - Selecionar 5-10 falantes com pelo menos 100 amostras cada
   - Organizar os dados na estrutura esperada

**Nota**: O download pode levar algum tempo dependendo da sua conexão.

### Estrutura de Dados Esperada

```
data/
└── voxceleb1/
    ├── id10001/  # Falante 1
    │   ├── audio1.wav
    │   ├── audio2.wav
    │   └── ...
    ├── id10002/  # Falante 2
    └── ...
```

## Pipeline de Execução

### 1. Análise Exploratória

Abra e execute o notebook:
```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

Este notebook irá:
- Carregar o dataset
- Visualizar estatísticas descritivas
- Explorar características dos áudios
- Gerar visualizações de espectrogramas

### 2. Extração de Features

```bash
jupyter notebook notebooks/02_feature_extraction.ipynb
```

Etapas:
- Pré-processamento de áudio (16kHz, mono, VAD, normalização)
- Extração de features acústicas:
  - **40 MFCCs** + deltas + delta-deltas
  - **4 features de pitch** (F0 via pYIN: mean, std, min, max)
  - **3 features espectrais** (centroid, rolloff, ZCR)
- Salvamento das features processadas

### 3. Treinamento Random Forest

Execute via notebook:
```bash
jupyter notebook notebooks/03_random_forest_baseline.ipynb
```

Ou via script:
```bash
python scripts/train_rf.py --config config/config.yaml --data data/processed/features_aggregated.pkl
```

Configuração:
- 150 árvores
- Profundidade máxima: 20
- Input: 188 features agregadas (47 features × 4 estatísticas)

### 4. Treinamento CNN

Execute via notebook:
```bash
jupyter notebook notebooks/04_cnn_model.ipynb
```

Ou via script:
```bash
python scripts/train_cnn.py --config config/config.yaml --data data/processed/features_sequential.h5
```

Arquitetura CNN:
- **Input**: (100 frames, 47 features)
- **3 Blocos Convolucionais**:
  - Conv1D (64, 128, 256 filtros) + BatchNorm + ReLU + MaxPool + Dropout
- **GlobalAveragePooling1D**
- **Dense(128)** + Dropout(0.5)
- **Dense(num_speakers, softmax)**
- ~180K parâmetros

### 5. Análise Comparativa

```bash
jupyter notebook notebooks/05_comparative_analysis.ipynb
```

Este notebook irá:
- Comparar métricas quantitativas (Accuracy, Precision, Recall, F1)
- Gerar matrizes de confusão
- Realizar testes estatísticos de significância
- Visualizar performance por falante
- Gerar gráficos para publicação

## Avaliação

### Métricas Implementadas

- **Accuracy**: Acurácia global
- **Precision**: Macro e Weighted
- **Recall**: Macro e Weighted
- **F1-Score**: Macro e Weighted
- **Confusion Matrix**: Matriz de confusão
- **Per-Speaker Accuracy**: Desempenho individual por falante
- **Statistical Tests**: Teste de Wilcoxon/t-test pareado

### Executar Avaliação

```bash
python scripts/evaluate_models.py \
    --rf-model models/random_forest_best.pkl \
    --cnn-model models/cnn_best.h5 \
    --test-data data/processed/test_features.h5 \
    --output results/
```

## Configuração

Todas as configurações do projeto estão centralizadas em `config/config.yaml`:

- **Dataset**: Número de falantes, diretórios
- **Preprocessing**: Taxa de amostragem, VAD, normalização
- **Features**: Parâmetros de MFCCs, pitch, spectral
- **Random Forest**: n_estimators, max_depth, etc.
- **CNN**: Arquitetura, filtros, dropout
- **Training**: Batch size, learning rate, callbacks
- **Evaluation**: Métricas, testes estatísticos

## Resultados Esperados

De acordo com a proposta, espera-se que:

- **Random Forest**: Baseline clássico com bom desempenho em condições controladas
- **CNN 1D**: Superior em capturar padrões temporais complexos
- **Comparação**: CNN deve superar RF em accuracy, especialmente com áudios mais desafiadores

### Formato de Resultados

```
results/
├── rf_metrics.json
├── cnn_metrics.json
├── confusion_matrix_rf.png
├── confusion_matrix_cnn.png
├── comparative_analysis.csv
└── statistical_tests.txt
```

## Reprodutibilidade

Para garantir resultados reproduzíveis:

1. **Seeds fixos**: Configurados em `config.yaml`
2. **Versões de dependências**: Especificadas em `requirements.txt`
3. **Configurações documentadas**: Todos os hiperparâmetros em `config.yaml`
4. **Pipeline determinístico**: Mesmo pré-processamento e splits de dados

## Troubleshooting

### Problema: "FFmpeg not found"
**Solução**: Instale o FFmpeg conforme instruções de instalação

### Problema: "Out of Memory" durante treinamento CNN
**Solução**: Reduza o `batch_size` em `config.yaml` (ex: 16 ao invés de 32)

### Problema: "No GPU found"
**Solução**: O código funciona em CPU, mas será mais lento. Para usar GPU, instale `tensorflow-gpu`

### Problema: Download do VoxCeleb1 falha
**Solução**: Baixe manualmente de [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) e organize conforme estrutura esperada

## Documentação Adicional

- **Metodologia Detalhada**: Ver `docs/metodologia.md`
- **Proposta Original**: Ver `PROPOSTA_DE_PESQUISA___Disciplina__BCC177___Redes_Neurais_VFinal.pdf`

## Contribuindo

Este é um projeto acadêmico. Para contribuir:

1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Referências

- **VoxCeleb1**: Nagrani, A., Chung, J. S., & Zisserman, A. (2017). VoxCeleb: A large-scale speaker identification dataset.
- **MFCCs**: Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition.
- **CNN for Audio**: Palaz, D., Collobert, R., & Doss, M. M. (2015). Analysis of CNN-based speech recognition system using raw speech as input.

## Licença

Este projeto é desenvolvido para fins acadêmicos como parte da disciplina BCC177 - Redes Neurais.

## Autores

- Disciplina: BCC177 - Redes Neurais
- Instituição: [Sua Instituição]
- Ano: 2024

## Contato

Para dúvidas ou sugestões, abra uma issue no repositório.

---

**Nota**: Este projeto implementa fielmente a proposta de pesquisa apresentada no arquivo PDF incluído no repositório.
