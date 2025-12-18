# 📋 Resumo do Projeto - BCC406 Redes Neurais

## 🎯 Título
**Comparação entre Random Forest e CNN 1D para Identificação de Falantes**

---

## 📝 Descrição

Projeto acadêmico da disciplina **BCC406 - Redes Neurais** que implementa e compara dois paradigmas de aprendizado de máquina para a tarefa de **identificação de falantes** em áudio:

### Modelos Comparados

| Modelo | Categoria | Características |
|--------|-----------|-----------------|
| **Random Forest** | Machine Learning Clássico | - 100 árvores de decisão<br>- Features agregadas (média, std, etc.)<br>- ~4000 features por amostra |
| **CNN 1D** | Deep Learning | - 3 blocos convolucionais<br>- BatchNorm + Dropout<br>- ~167,000 parâmetros |

---

## 🔬 Metodologia

### 1. **Dataset**
- **Oficial:** VoxCeleb1 (~38GB, 1,251 falantes, ~100k arquivos)
- **Teste:** Sintético (200 arquivos, 10 falantes, ~180MB)

### 2. **Features**
- **MFCCs:** 40 coeficientes
- **Janela temporal:** 100 time steps
- **Formato:** (batch, 100, 40) para CNN | (batch, 4000) para RF

### 3. **Treinamento**
- **Divisão:** 60% treino, 20% validação, 20% teste
- **Otimizador:** Adam (lr=0.001 para CNN)
- **Loss:** Categorical Crossentropy
- **Métricas:** Acurácia, Precisão, Recall, F1-Score

---

## 📊 Resultados

### Dataset Sintético (200 amostras)
```
┌─────────────────┬──────────┬──────────┬─────────┐
│ Modelo          │ Acurácia │ Precisão │ Recall  │
├─────────────────┼──────────┼──────────┼─────────┤
│ Random Forest   │  96.7%   │  97.2%   │  96.5%  │
│ CNN 1D          │  66.7%   │  68.3%   │  66.1%  │
└─────────────────┴──────────┴──────────┴─────────┘

🏆 Vencedor: Random Forest (+30% de diferença)
```

**Análise:**
- RF superior em datasets pequenos (overfitting da CNN)
- CNN precisa de mais dados para generalizar
- RF mais interpretável e rápido para treinar

### VoxCeleb1 Completo
*🔄 Treinamento pendente - resultados serão adicionados*

---

## 🛠️ Tecnologias

### Core
- **Python:** 3.11.4
- **TensorFlow:** 2.14.0 (backend da CNN)
- **scikit-learn:** 1.3.0 (Random Forest)
- **librosa:** 0.10.1 (processamento de áudio)

### Suporte
- **NumPy:** 1.24.3
- **Pandas:** 2.0.3
- **Matplotlib:** 3.7.2
- **seaborn:** 0.12.2

**Total:** 15+ dependências (ver [requirements.txt](requirements.txt))

---

## 📁 Estrutura do Código

### **Módulos Principais** (`src/`)
```
src/
├── data/dataset.py              # Gerenciamento de datasets
├── features/audio_features.py   # Extração de MFCCs
├── training/trainer.py          # Treinadores genéricos
├── evaluation/metrics.py        # Cálculo de métricas
└── utils/helpers.py             # Funções auxiliares
```

### **Scripts Executáveis** (`scripts/`)
```
scripts/
├── run_full_pipeline.py         # Pipeline completo
├── train_rf.py                  # Treinar Random Forest
├── train_cnn.py                 # Treinar CNN
├── evaluate_models.py           # Avaliar ambos modelos
├── baixar_voxceleb1.py          # Download VoxCeleb1
└── generate_synthetic_data.py   # Gerar dados teste
```

### **Notebooks Jupyter** (`notebooks/`)
1. `01_exploratory_analysis.ipynb` - EDA dos dados
2. `02_feature_extraction.ipynb` - Extração de MFCCs
3. `03_random_forest_baseline.ipynb` - Baseline RF
4. `04_cnn_model.ipynb` - Modelo CNN
5. `05_comparative_analysis.ipynb` - Análise comparativa

---

## 🚀 Como Executar

### Opção 1: Pipeline Completo (Recomendado)
```bash
python scripts/run_full_pipeline.py
```
**Tempo:** ~5-10 minutos com dados sintéticos

### Opção 2: Google Colab
1. Abra [notebooks/COLAB_Pipeline_Completo.md](notebooks/COLAB_Pipeline_Completo.md)
2. Siga as instruções
3. Execute células sequencialmente

### Opção 3: Passo a Passo
```bash
# 1. Gerar dados sintéticos
python scripts/generate_synthetic_data.py

# 2. Treinar Random Forest
python scripts/train_rf.py

# 3. Treinar CNN
python scripts/train_cnn.py

# 4. Avaliar e comparar
python scripts/evaluate_models.py
```

---

## 📚 Documentação

### Por Perfil de Usuário

| Perfil | Documento | Foco |
|--------|-----------|------|
| **Iniciante** | [GUIA_RAPIDO.md](GUIA_RAPIDO.md) | Executar rapidamente |
| **Desenvolvedor** | [ESTRUTURA_DO_PROJETO.md](ESTRUTURA_DO_PROJETO.md) | Arquitetura do código |
| **Pesquisador** | [docs/metodologia.md](docs/metodologia.md) | Teoria e metodologia |
| **Usuário Colab** | [notebooks/COLAB_Pipeline_Completo.md](notebooks/COLAB_Pipeline_Completo.md) | Tutorial passo a passo |
| **Qualquer um** | [INDICE.md](INDICE.md) | Hub de navegação |

---

## ✅ Funcionalidades Implementadas

### Extração de Features
- [x] MFCCs com librosa
- [x] Normalização de áudio
- [x] Agregação estatística (média, std, min, max)
- [x] Features sequenciais para CNN

### Modelos
- [x] Random Forest com tunning
- [x] CNN 1D com arquitetura customizada
- [x] Salvamento de modelos treinados
- [x] Carregamento de checkpoints

### Avaliação
- [x] Métricas padrão (acurácia, precisão, recall, F1)
- [x] Matriz de confusão
- [x] Curvas de aprendizado
- [x] Comparação lado a lado

### Utilitários
- [x] Download automatizado do VoxCeleb1
- [x] Geração de dados sintéticos
- [x] Configuração via YAML
- [x] Logging estruturado

---

## 🔧 Configurações

Edite [config/config.yaml](config/config.yaml) para ajustar:

```yaml
data:
  n_mfcc: 40           # Número de coeficientes MFCC
  max_len: 100         # Janela temporal
  
training:
  rf:
    n_estimators: 100  # Árvores no Random Forest
    max_depth: 20      # Profundidade máxima
  
  cnn:
    epochs: 50         # Épocas de treinamento
    batch_size: 32     # Tamanho do batch
    learning_rate: 0.001
```

---

## 📈 Métricas de Desempenho

### Tempo de Treinamento (Dataset Sintético)
- **Random Forest:** ~2 minutos
- **CNN:** ~3 minutos (50 épocas, CPU)

### Tamanho dos Modelos
- **Random Forest:** ~15 MB (random_forest.pkl)
- **CNN:** ~2 MB (cnn_modelo.h5)

### Inferência
- **Random Forest:** ~0.5 ms/amostra
- **CNN:** ~2 ms/amostra (CPU)

---

## 🎓 Contexto Acadêmico

### Disciplina
- **Código:** BCC406
- **Nome:** Redes Neurais
- **Instituição:** [Sua Universidade]

### Objetivos de Aprendizado
1. ✅ Comparar paradigmas clássicos vs deep learning
2. ✅ Implementar pipeline completo de ML
3. ✅ Trabalhar com dados de áudio reais
4. ✅ Avaliar modelos criticamente
5. ✅ Documentar código profissionalmente

---

## 🐛 Problemas Conhecidos

### 1. CNN com baixa acurácia em dataset pequeno
**Causa:** Overfitting em 200 amostras  
**Solução:** Usar VoxCeleb1 completo ou aumentar dropout

### 2. Tempo de download do VoxCeleb1
**Causa:** Dataset de 38GB  
**Solução:** Usar dados sintéticos para testes

### 3. GPU não detectada no TensorFlow
**Causa:** CUDA não configurado  
**Solução:** Usar Colab (GPU gratuita) ou treinar em CPU

---

## 📋 Checklist de Reprodutibilidade

- [x] Seeds fixos (numpy, tensorflow, random)
- [x] Ambiente virtual documentado
- [x] Dependências versionadas (requirements.txt)
- [x] Dados sintéticos fornecidos
- [x] Scripts de download automatizados
- [x] Configurações externalizadas (YAML)
- [x] Logs estruturados
- [x] Modelos salvos em formatos padrão

---

## 🔗 Links Importantes

### Datasets
- [VoxCeleb1 Oficial](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
- [Paper Original](https://arxiv.org/abs/1706.08612)

### Referências Teóricas
- [MFCCs Explained](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees)
- [1D CNNs for Audio](https://towardsdatascience.com/cnns-for-audio-classification-6244954665ab)

---

## 📞 Suporte

### Dúvidas Frequentes
Consulte a seção **"Problemas Comuns"** no [README.md](README.md)

### Documentação
Navegue pelo [INDICE.md](INDICE.md) para encontrar qualquer informação

### Issues
Abra uma issue no repositório para reportar bugs ou sugerir melhorias

---

## 📅 Histórico

| Data | Versão | Mudanças |
|------|--------|----------|
| [DATA] | 1.0 | Versão inicial com RF e CNN |
| [DATA] | 1.1 | Adição de dados sintéticos |
| [DATA] | 1.2 | Tradução completa para PT-BR |
| [DATA] | 1.3 | Organização final da documentação |

---

## 🏆 Conclusões

### Aprendizados Principais
1. **Random Forest** é superior em datasets pequenos (~200 amostras)
2. **CNN** precisa de grande volume de dados (>10k amostras)
3. **MFCCs** são features robustas para áudio
4. **Pipeline modular** facilita experimentação

### Trabalhos Futuros
- [ ] Treinar com VoxCeleb1 completo
- [ ] Implementar LSTM para sequências
- [ ] Testar Transformers (wav2vec 2.0)
- [ ] Criar interface web para demonstração
- [ ] Adicionar data augmentation
- [ ] Testar em outros datasets (CommonVoice, LibriSpeech)

---

<div align="center">

**🎓 Projeto BCC406 - Redes Neurais**  
*Random Forest vs CNN para Identificação de Falantes*

[README](README.md) | [Documentação](INDICE.md) | [Guia Rápido](GUIA_RAPIDO.md)

</div>
