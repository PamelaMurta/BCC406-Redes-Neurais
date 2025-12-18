# 📊 Guia Rápido de Execução - RF vs CNN para Identificação de Falantes

## 🎯 Resumo Executivo

Este projeto compara **Random Forest** e **CNN 1D** para identificação de falantes usando o dataset VoxCeleb1.

**Resultado do Experimento (Dataset Sintético - 200 amostras, 10 falantes):**
- 🏆 **Vencedor**: Random Forest (96.7% de acurácia)
- 🥈 **Segundo**: CNN 1D (66.7% de acurácia)

---

## ⚡ Execução Rápida (3 Opções)

### Opção 1: Google Colab (Mais Fácil) ⭐

```
1. Acesse: notebooks/COLAB_Pipeline_Completo.md
2. Copie o código para um novo notebook no Colab
3. Execute célula por célula
4. Aguarde ~30-60 minutos
5. Pronto! Resultados prontos
```

**Vantagens:**
- ✅ Sem instalação local
- ✅ GPU grátis
- ✅ Ambiente pré-configurado

---

### Opção 2: Script Único (Local)

Se você já tem Python e os dados:

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Executar pipeline completo
python scripts/run_full_pipeline.py

# 3. Ver resultados em results/
```

**Tempo**: ~10-30 minutos (dependendo do tamanho do dataset)

---

### Opção 3: Passo a Passo (Notebooks)

Para análise detalhada:

```bash
# 1. Análise exploratória
jupyter notebook notebooks/01_exploratory_analysis.ipynb

# 2. Extração de features
python scripts/run_01_extract_features.py

# 3. Treinar Random Forest
python scripts/train_rf.py

# 4. Treinar CNN
python scripts/train_cnn.py

# 5. Comparar resultados
python scripts/evaluate_models.py
```

---

## 📁 Estrutura de Arquivos Principais

```
📦 Projeto
├── 📓 notebooks/
│   ├── COLAB_Pipeline_Completo.md    ⭐ Comece aqui (Colab)
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_random_forest_baseline.ipynb
│   ├── 04_cnn_model.ipynb
│   └── 05_comparative_analysis.ipynb
│
├── 🔧 scripts/
│   ├── baixar_voxceleb1.py          ⭐ Download automático do dataset
│   ├── run_full_pipeline.py         ⭐ Execução completa
│   ├── generate_synthetic_data.py    # Criar dados de teste
│   └── test_notebook01.py            # Testar instalação
│
├── 📚 src/
│   ├── data/dataset.py               # Gerenciamento de dados
│   ├── features/audio_features.py    # Extração de features
│   ├── models/                       # Modelos (RF e CNN)
│   └── utils/helpers.py              # Funções auxiliares
│
└── ⚙️ config/
    └── config.yaml                   # Configurações centralizadas
```

---

## 🎓 Resultados Obtidos

### Com Dataset Sintético (200 amostras)

| Métrica | Random Forest | CNN 1D |
|---------|--------------|--------|
| **Acurácia Treino** | 100.0% | 74.3% |
| **Acurácia Validação** | 96.7% | 63.3% |
| **Acurácia Teste** | **96.7%** | 66.7% |
| **Tempo Treino** | < 1 min | 5-10 min |
| **Parâmetros** | ~180K | ~167K |

**Conclusão**: Para datasets pequenos (<1000 amostras), Random Forest supera CNN devido à maior eficiência com poucos dados.

### Expectativa com VoxCeleb1 Completo (100K+ amostras)

Com mais dados, esperamos que:
- 📈 CNN alcance **85-95%** de acurácia
- 📊 RF se mantenha em **80-90%** de acurácia
- 🏆 CNN supere RF devido à capacidade de aprender padrões complexos

---

## 🔍 Checklist de Execução

### Antes de Começar

- [ ] Python 3.8+ instalado
- [ ] 40GB de espaço livre (dataset completo) ou 10GB (teste)
- [ ] Conexão estável de internet (para download)
- [ ] (Opcional) Conta no Google Colab

### Execução Local

- [ ] Ambiente virtual criado
- [ ] Dependências instaladas (`pip install -r requirements.txt`)
- [ ] Dataset baixado e extraído
- [ ] Configurações ajustadas em `config/config.yaml`

### Execução no Colab

- [ ] Notebook copiado para Colab
- [ ] Runtime com GPU selecionado
- [ ] Dataset disponível (Drive ou download automático)

---

## ❓ Perguntas Frequentes

### P: Quanto tempo leva para executar?

**R**: 
- Colab (teste): 30-60 min
- Local (sintético): 10-15 min
- Local (completo): 2-4 horas

### P: Preciso de GPU?

**R**: 
- Random Forest: Não
- CNN: Recomendado (treina 10-20x mais rápido)
- Colab fornece GPU grátis!

### P: Posso usar meu próprio dataset de áudio?

**R**: Sim! Basta organizar em:
```
data/raw/
  ├── falante1/
  │   ├── audio1.wav
  │   └── audio2.wav
  ├── falante2/
  └── ...
```

### P: Os modelos já estão treinados?

**R**: Não, você treina do zero. Mas o processo é automatizado!

### P: Como citar este projeto?

**R**:
```bibtex
@misc{rf-vs-cnn-speaker-id-2025,
  author = {Projeto BCC406},
  title = {Comparação RF vs CNN para Identificação de Falantes},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/PamelaMurta/BCC406-Redes-Neurais}
}
```

---

## 🆘 Solução de Problemas

### Erro: "Module not found"
```bash
pip install -r requirements.txt --upgrade
```

### Erro: "Out of memory"
```python
# Reduzir batch size em config/config.yaml
training:
  batch_size: 16  # Era 32
```

### Erro: "Dataset not found"
```bash
# Verificar estrutura
ls -R data/raw/

# Recriar dados sintéticos
python scripts/generate_synthetic_data.py
```

### Erro no Colab: "Runtime disconnected"
- Use dataset menor (teste apenas)
- Salve checkpoints frequentemente
- Execute em horários de menor uso

---

## 📞 Suporte

- 📧 **Email**: [seu-email]
- 🐛 **Issues**: [GitHub Issues](https://github.com/PamelaMurta/BCC406-Redes-Neurais/issues)
- 📚 **Docs**: [docs/metodologia.md](docs/metodologia.md)

---

## 📜 Licença

MIT License - Uso livre para fins acadêmicos e educacionais

---

**Última atualização**: Dezembro 2025  
**Versão**: 1.0  
**Status**: ✅ Produção
