# Metodologia Detalhada

## Projeto: Comparação RF vs CNN para Identificação de Falantes

Este documento fornece detalhes da metodologia implementada neste projeto, baseado na proposta de pesquisa "COMPARAÇÃO ENTRE RANDOM FOREST E REDES NEURAIS CONVOLUCIONAIS PARA IDENTIFICAÇÃO DE FALANTES EM CONDIÇÕES NÃO CONTROLADAS".

---

## 1. Visão Geral

O projeto implementa um pipeline completo de identificação de falantes comparando dois paradigmas:

- **Random Forest (RF)**: Método clássico de Machine Learning usando features agregadas
- **CNN 1D**: Método de Deep Learning usando features sequenciais

---

## 2. Dataset: VoxCeleb1

### 2.1 Características
- **Dataset**: VoxCeleb1 development set
- **Origem**: Vídeos do YouTube
- **Condições**: Não controladas (ruído, reverberação, variação de qualidade)
- **Conteúdo**: Fala natural em diferentes contextos

### 2.2 Seleção de Falantes
- **Regime**: Poucos falantes (5-10 falantes)
- **Critério**: Mínimo de 100 amostras por falante
- **Distribuição**: 70% treino, 15% validação, 15% teste

### 2.3 Motivação
Avaliar robustez dos métodos em condições reais com conjunto reduzido de falantes.

---

## 3. Pré-processamento de Áudio (Seção 3.2)

### 3.1 Conversão e Padronização
```python
- Sample Rate: 16 kHz
- Canais: Mono (1 canal)
- Formato: WAV
```

**Implementação**: `src/data/preprocessing.py::convert_to_wav()`

### 3.2 Voice Activity Detection (VAD)
```python
- Método: Energy-based (librosa.effects.split)
- Threshold: 20 dB abaixo do máximo
- Frame length: 2048 samples
- Hop length: 512 samples
```

**Objetivo**: Remover silêncios e ruídos de fundo.

**Implementação**: `src/data/preprocessing.py::apply_vad()`

### 3.3 Normalização de Amplitude
```python
- Método: Normalização linear
- Amplitude máxima: 1.0
```

**Objetivo**: Padronizar níveis de volume entre diferentes gravações.

**Implementação**: `src/data/preprocessing.py::normalize_amplitude()`

### 3.4 Padding/Truncamento
```python
- Método: Pad com zeros ou truncar do centro
- Aplicado em nível de features (Tmax = 100 frames)
```

**Implementação**: `src/features/audio_features.py::pad_features_to_max_length()`

---

## 4. Extração de Features (Seção 3.3)

### 4.1 MFCCs (Mel-Frequency Cepstral Coefficients)
```python
Parâmetros:
- n_mfcc: 40
- n_fft: 2048
- hop_length: 512
- n_mels: 128
- fmin: 0 Hz
- fmax: 8000 Hz
```

**Output**: 40 coeficientes por frame temporal

**Significado**: Representação compacta do espectro de frequências, modelando características do trato vocal.

**Implementação**: `src/features/audio_features.py::extract_mfcc()`

### 4.2 Pitch Features (F0)
```python
Método: pYIN (Probabilistic YIN)
Parâmetros:
- fmin: 80 Hz
- fmax: 400 Hz
- frame_length: 2048
```

**Output**: 4 estatísticas (mean, std, min, max) + contorno temporal

**Significado**: Frequência fundamental da voz, característica importante para identificação de falantes.

**Implementação**: `src/features/audio_features.py::extract_pitch()`

### 4.3 Spectral Features
```python
1. Spectral Centroid: Centro de massa do espectro
2. Spectral Rolloff: Frequência abaixo da qual está 85% da energia
3. Zero Crossing Rate: Taxa de mudanças de sinal
```

**Output**: 3 features por frame temporal

**Significado**: Características timbrais e texturais do sinal.

**Implementação**: `src/features/audio_features.py::extract_spectral_features()`

### 4.4 Feature Vector Final

#### Para CNN (Sequential):
```
Shape: (Tmax, F) = (100, 47)
- 40 MFCCs
- 1 Pitch contour
- 3 Spectral features
- 3 Pitch statistics (replicadas temporalmente)
Total: 47 features por frame
```

#### Para Random Forest (Aggregated):
```
Shape: (188,)
- 47 features × 4 estatísticas (mean, std, min, max)
Total: 188 features agregadas
```

**Implementação**: 
- Sequential: `src/features/audio_features.py::extract_all_features()`
- Aggregated: `src/features/feature_aggregation.py::aggregate_features()`

---

## 5. Modelo Random Forest (Seção 3.4.1)

### 5.1 Configuração
```python
Parâmetros:
- n_estimators: 150 árvores
- max_depth: 20
- criterion: Gini impurity
- min_samples_split: 2
- min_samples_leaf: 1
- max_features: sqrt(n_features)
- bootstrap: True
- n_jobs: -1 (paralelo)
```

### 5.2 Input/Output
```
Input: (N, 188) - N amostras, 188 features agregadas
Output: (N,) - Probabilidades para cada falante
```

### 5.3 Vantagens
- Treinamento rápido
- Interpretável (feature importance)
- Robusto a overfitting
- Não requer GPU

### 5.4 Limitações
- Perde informação temporal
- Features manualmente projetadas
- Capacidade limitada para padrões complexos

**Implementação**: `src/models/random_forest.py`

---

## 6. Modelo CNN 1D (Seção 3.4.2)

### 6.1 Arquitetura

```
Input: (100, 47)
    ↓
┌─────────────────────┐
│ Conv1D Block 1      │
│ - Conv1D(64, k=3)   │
│ - BatchNorm         │
│ - ReLU              │
│ - MaxPool(2)        │
│ - Dropout(0.3)      │
└─────────────────────┘
    ↓ (50, 64)
┌─────────────────────┐
│ Conv1D Block 2      │
│ - Conv1D(128, k=3)  │
│ - BatchNorm         │
│ - ReLU              │
│ - MaxPool(2)        │
│ - Dropout(0.3)      │
└─────────────────────┘
    ↓ (25, 128)
┌─────────────────────┐
│ Conv1D Block 3      │
│ - Conv1D(256, k=3)  │
│ - BatchNorm         │
│ - ReLU              │
│ - MaxPool(2)        │
│ - Dropout(0.3)      │
└─────────────────────┘
    ↓ (12, 256)
┌─────────────────────┐
│ GlobalAvgPool1D     │
└─────────────────────┘
    ↓ (256,)
┌─────────────────────┐
│ Dense(128) + ReLU   │
│ Dropout(0.5)        │
└─────────────────────┘
    ↓ (128,)
┌─────────────────────┐
│ Dense(num_speakers) │
│ Softmax             │
└─────────────────────┘
    ↓ (num_speakers,)
Output: Probabilidades
```

### 6.2 Parâmetros
- **Total**: ~180K parâmetros treináveis
- **Filtros**: Progressão 64 → 128 → 256
- **Kernel size**: 3 (captura contexto local)
- **Padding**: Same (mantém dimensão temporal)

### 6.3 Componentes Importantes

#### BatchNormalization
- Acelera convergência
- Reduz internal covariate shift
- Efeito regularizador

#### Dropout
- 0.3 em camadas convolucionais
- 0.5 em camada densa
- Previne overfitting

#### GlobalAveragePooling
- Reduz dimensionalidade
- Robusto a variações de comprimento
- Menos parâmetros que Flatten

### 6.4 Vantagens
- Aprende features automaticamente
- Captura padrões temporais
- Alta capacidade de representação
- Estado da arte em tarefas de áudio

### 6.5 Limitações
- Requer mais dados
- Treinamento mais lento
- Necessita GPU para eficiência
- Menos interpretável

**Implementação**: `src/models/cnn_1d.py`

---

## 7. Treinamento (Seção 3.5)

### 7.1 Configuração CNN

```python
Optimizer: Adam
- Learning rate: 0.001
- beta_1: 0.9
- beta_2: 0.999
- epsilon: 1e-7

Loss: Categorical Crossentropy
Batch size: 32
Max epochs: 100
```

### 7.2 Callbacks

#### Early Stopping
```python
- Monitor: val_loss
- Patience: 15 epochs
- Restore best weights: True
```

#### ReduceLROnPlateau
```python
- Monitor: val_loss
- Factor: 0.5
- Patience: 5 epochs
- Min LR: 1e-6
```

#### ModelCheckpoint
```python
- Monitor: val_accuracy
- Save best only: True
```

### 7.3 Data Augmentation
Não implementado na versão base, mas pode incluir:
- Time stretching
- Pitch shifting
- Adding noise
- SpecAugment

**Implementação**: `src/training/trainer.py`, `src/training/callbacks.py`

---

## 8. Avaliação (Seção 3.6)

### 8.1 Métricas Principais

#### Accuracy
```
Accuracy = Corretos / Total
```

#### Precision (Macro e Weighted)
```
Precision = TP / (TP + FP)
Macro: média simples entre classes
Weighted: média ponderada por suporte
```

#### Recall (Macro e Weighted)
```
Recall = TP / (TP + FN)
```

#### F1-Score (Macro e Weighted)
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### 8.2 Análises Complementares

#### Confusion Matrix
- Visualiza erros entre pares de falantes
- Identifica confusões sistemáticas
- Normalizada por linha (classe verdadeira)

#### Per-Speaker Accuracy
- Accuracy individual de cada falante
- Identifica falantes difíceis
- Útil para datasets desbalanceados

### 8.3 Testes Estatísticos

#### Wilcoxon Signed-Rank Test
```python
- Teste não-paramétrico
- Compara médias de dois modelos
- H0: Não há diferença significativa
- α = 0.05
```

**Uso**: Comparar RF vs CNN

**Implementação**: `src/evaluation/metrics.py`, `src/evaluation/visualization.py`

---

## 9. Pipeline Completo

### 9.1 Workflow de Treinamento

```
1. Download VoxCeleb1
   ↓
2. Selecionar falantes (5-10)
   ↓
3. Split train/val/test (70/15/15)
   ↓
4. Para cada áudio:
   - Carregar
   - Pré-processar (VAD, normalização)
   - Extrair features
   ↓
5. Salvar features
   - Sequential: Para CNN
   - Aggregated: Para RF
   ↓
6a. Treinar Random Forest        6b. Treinar CNN
   - 150 árvores                    - 100 epochs max
   - Features agregadas             - Features sequenciais
   - Rápido (~minutos)              - Lento (~horas)
   ↓                                 ↓
7. Avaliar ambos no test set
   ↓
8. Comparação quantitativa
   - Métricas
   - Matrizes de confusão
   - Testes estatísticos
   ↓
9. Análise e visualizações
```

### 9.2 Reprodutibilidade

#### Seeds Fixos
```python
numpy.random.seed(42)
tensorflow.random.set_seed(42)
sklearn random_state=42
```

#### Versionamento
- Dependências fixas em requirements.txt
- Configurações em config.yaml
- Logs completos salvos

---

## 10. Resultados Esperados

### 10.1 Hipóteses

1. **CNN > RF**: CNN deve superar RF em accuracy
2. **Temporal matters**: Features sequenciais capturam mais informação
3. **Trade-off**: RF mais rápido, CNN mais preciso

### 10.2 Análises

- Comparação quantitativa (Tabela 3 da proposta)
- Análise de erros (confusion matrices)
- Feature importance (RF)
- Robustez a qualidade de áudio

---

## 11. Referências Técnicas

### Bibliotecas Principais
- **librosa**: Processamento de áudio
- **TensorFlow/Keras**: Deep Learning
- **scikit-learn**: Machine Learning clássico
- **NumPy/Pandas**: Manipulação de dados
- **Matplotlib/Seaborn**: Visualização

### Papers Relacionados
1. Nagrani et al. (2017) - VoxCeleb dataset
2. Davis & Mermelstein (1980) - MFCCs
3. Mauch & Dixon (2014) - pYIN algorithm
4. Snyder et al. (2018) - x-vectors for speaker recognition

---

## 12. Extensões Futuras

### Possíveis Melhorias
1. **Mais falantes**: Escalar para regime de muitos falantes
2. **Data augmentation**: Aumentar robustez
3. **Arquiteturas modernas**: ResNet, Attention, Transformers
4. **Transfer learning**: Usar modelos pré-treinados
5. **Multi-task learning**: Combinar com outras tarefas (emotion, gender)
6. **Ensemble**: Combinar RF + CNN

---

**Última atualização**: 2024
**Disciplina**: BCC177 - Redes Neurais
**Projeto**: Identificação de Falantes - RF vs CNN
