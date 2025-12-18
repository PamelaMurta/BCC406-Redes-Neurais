# 📚 Índice de Documentação - Projeto RF vs CNN

## 🎯 Comece Aqui

Dependendo do seu objetivo, escolha um dos caminhos abaixo:

---

## 🚀 Para Executar o Projeto

### 1️⃣ Execução Rápida (Recomendado)

```
📖 Leia: GUIA_RAPIDO.md
   ↓
🚀 Execute: scripts/run_full_pipeline.py
   ↓
📊 Veja resultados: results/
```

**Tempo**: ~10-15 minutos  
**Requisito**: Python 3.8+ instalado

---

### 2️⃣ Execução no Google Colab

```
📖 Abra: notebooks/COLAB_Pipeline_Completo.md
   ↓
📋 Copie para novo notebook Colab
   ↓
▶️ Execute células sequencialmente
   ↓
💾 Baixe resultados
```

**Tempo**: ~30-60 minutos  
**Requisito**: Conta Google

---

## 📖 Para Entender o Projeto

### Documentação por Nível

#### 🟢 Iniciante

1. **README.md** - Visão geral e instalação
2. **GUIA_RAPIDO.md** - Início rápido com 3 opções
3. **ESTRUTURA_DO_PROJETO.md** - Organização dos arquivos

#### 🟡 Intermediário

1. **IMPLEMENTATION_SUMMARY.md** - Resumo técnico
2. **notebooks/01_exploratory_analysis.ipynb** - Análise de dados
3. **notebooks/05_comparative_analysis.ipynb** - Comparação de modelos

#### 🔴 Avançado

1. **docs/metodologia.md** - Metodologia completa
2. **src/** - Código fonte detalhado
3. **config/config.yaml** - Todos os parâmetros

---

## 🎓 Por Objetivo

### Quero Aprender sobre o Projeto

```
1. README.md                              # Visão geral
2. IMPLEMENTATION_SUMMARY.md              # O que foi feito
3. docs/metodologia.md                    # Como foi feito
4. notebooks/01_exploratory_analysis.ipynb # Dados
5. notebooks/05_comparative_analysis.ipynb # Resultados
```

### Quero Replicar os Experimentos

```
1. GUIA_RAPIDO.md                         # Como executar
2. scripts/baixar_voxceleb1.py           # Obter dados
3. scripts/run_full_pipeline.py          # Rodar tudo
4. config/config.yaml                     # Ajustar parâmetros
```

### Quero Modificar o Código

```
1. ESTRUTURA_DO_PROJETO.md               # Organização
2. src/data/dataset.py                   # Gerenciar dados
3. src/features/audio_features.py        # Features
4. src/training/trainer.py               # Treinamento
5. config/config.yaml                     # Configurações
```

### Quero Usar no Colab

```
1. notebooks/COLAB_Pipeline_Completo.md  # Tutorial completo
2. GUIA_RAPIDO.md                        # FAQ e troubleshooting
```

---

## 📊 Documentos por Tipo

### 📘 Guias e Tutoriais

| Documento | Descrição | Tempo Leitura |
|-----------|-----------|---------------|
| `README.md` | Documentação principal | 10 min |
| `GUIA_RAPIDO.md` | Início rápido | 5 min |
| `ESTRUTURA_DO_PROJETO.md` | Organização | 5 min |
| `notebooks/COLAB_Pipeline_Completo.md` | Tutorial Colab | 60 min (executando) |

### 📗 Documentação Técnica

| Documento | Descrição | Tempo Leitura |
|-----------|-----------|---------------|
| `IMPLEMENTATION_SUMMARY.md` | Resumo implementação | 15 min |
| `docs/metodologia.md` | Metodologia completa | 30 min |
| `config/config.yaml` | Parâmetros | 5 min |

### 📙 Código e Scripts

| Arquivo | Descrição | Tipo |
|---------|-----------|------|
| `scripts/run_full_pipeline.py` | Pipeline completo | Script |
| `scripts/baixar_voxceleb1.py` | Download dataset | Script |
| `src/data/dataset.py` | Gerenciamento dados | Módulo |
| `src/features/audio_features.py` | Extração features | Módulo |

### 📕 Notebooks

| Notebook | Descrição | Tempo |
|----------|-----------|-------|
| `01_exploratory_analysis.ipynb` | Análise exploratória | 15 min |
| `02_feature_extraction.ipynb` | Extração features | 30 min |
| `03_random_forest_baseline.ipynb` | Treinar RF | 10 min |
| `04_cnn_model.ipynb` | Treinar CNN | 30 min |
| `05_comparative_analysis.ipynb` | Comparação | 15 min |

---

## 🔍 Busca Rápida

### Por Palavra-Chave

**Instalação**: `README.md` → Seção "Instalação"  
**Dataset**: `scripts/baixar_voxceleb1.py` ou `GUIA_RAPIDO.md`  
**Configuração**: `config/config.yaml`  
**Colab**: `notebooks/COLAB_Pipeline_Completo.md`  
**Resultados**: `GUIA_RAPIDO.md` → "Resultados Obtidos"  
**Erros**: `GUIA_RAPIDO.md` → "Solução de Problemas"  
**Pipeline**: `scripts/run_full_pipeline.py`  
**Código Fonte**: `src/`  
**Features**: `src/features/audio_features.py`  
**Modelos**: `src/models/` (código) ou `models/` (treinados)  

---

## 📱 Acesso Rápido por Dispositivo

### 💻 Desktop/Notebook

**Recomendação**: Instalação local
```
1. Clone repositório
2. Instale dependências
3. Execute: python scripts/run_full_pipeline.py
```

### 📱 Tablet/Mobile

**Recomendação**: Google Colab
```
1. Abra: notebooks/COLAB_Pipeline_Completo.md
2. Use no Colab
```

### ☁️ Cloud/Servidor

**Recomendação**: Docker ou Script
```
1. Clone repositório
2. Configure ambiente
3. Execute pipeline
```

---

## 🗺️ Mapa de Dependências

```
README.md
├─ GUIA_RAPIDO.md
│  └─ scripts/run_full_pipeline.py
│     ├─ config/config.yaml
│     ├─ src/data/dataset.py
│     ├─ src/features/audio_features.py
│     └─ src/training/trainer.py
│
├─ ESTRUTURA_DO_PROJETO.md
│
├─ notebooks/COLAB_Pipeline_Completo.md
│  └─ scripts/baixar_voxceleb1.py
│
└─ IMPLEMENTATION_SUMMARY.md
   └─ docs/metodologia.md
```

---

## ✅ Checklist de Documentos

### Para Começar

- [ ] Li `README.md`
- [ ] Li `GUIA_RAPIDO.md`
- [ ] Entendi `ESTRUTURA_DO_PROJETO.md`

### Para Executar

- [ ] Instalei dependências (`requirements.txt`)
- [ ] Configurei ambiente (Python 3.8+)
- [ ] Baixei ou gerei dados
- [ ] Ajustei `config/config.yaml` (se necessário)

### Para Entender

- [ ] Li `IMPLEMENTATION_SUMMARY.md`
- [ ] Explorei `notebooks/01_exploratory_analysis.ipynb`
- [ ] Revisei `docs/metodologia.md`

### Para Modificar

- [ ] Entendi `src/` estrutura
- [ ] Revisei código fonte
- [ ] Testei modificações

---

## 📞 Ainda com Dúvidas?

1. **Procure no FAQ**: `GUIA_RAPIDO.md` → Seção "Perguntas Frequentes"
2. **Veja exemplos**: `notebooks/` → Notebooks interativos
3. **Revise código**: `src/` → Código fonte comentado
4. **Abra issue**: GitHub Issues

---

## 🎯 Caminho Recomendado (Primeira Vez)

```
DIA 1 (30 min)
├─ Leia README.md (10 min)
├─ Leia GUIA_RAPIDO.md (10 min)
└─ Configure ambiente (10 min)

DIA 2 (60 min)
├─ Gere dados sintéticos (5 min)
├─ Execute pipeline (10 min)
└─ Explore notebooks (45 min)

DIA 3 (120 min)
├─ Baixe VoxCeleb1 (30 min)
├─ Execute com dados reais (60 min)
└─ Analise resultados (30 min)
```

---

**Última atualização**: Dezembro 2025  
**Versão**: 1.0  

💡 **Dica**: Comece sempre pelo `GUIA_RAPIDO.md` para economizar tempo!
