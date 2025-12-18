# Notebook Completo para Google Colab - RF vs CNN para Identificação de Falantes

Este notebook implementa o pipeline completo de treinamento e comparação entre Random Forest e CNN 1D para identificação de falantes usando o dataset VoxCeleb1.

## 📋 Índice
1. Configuração do Ambiente
2. Download do Dataset VoxCeleb1
3. Análise Exploratória
4. Extração de Features
5. Treinamento Random Forest
6. Treinamento CNN 1D
7. Análise Comparativa
8. Resultados e Conclusões

---

## 1️⃣ Configuração do Ambiente

```python
# Verificar GPU
!nvidia-smi

# Instalar dependências
!pip install librosa soundfile pydub scikit-learn tensorflow matplotlib seaborn tqdm pyyaml h5py joblib wget

# Clonar repositório
!git clone https://github.com/seu-usuario/BCC406-Redes-Neurais.git
%cd BCC406-Redes-Neurais

# Importar bibliotecas
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import pickle

print("✓ Ambiente configurado")
print(f"TensorFlow: {tf.__version__}")
print(f"GPU disponível: {tf.config.list_physical_devices('GPU')}")
```

---

## 2️⃣ Download do Dataset VoxCeleb1

```python
# Opção 1: Download do conjunto de teste (menor, ~5GB)
!python scripts/baixar_voxceleb1.py

# OU

# Opção 2: Upload manual do Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copiar do Drive
!cp -r /content/drive/MyDrive/VoxCeleb1 data/raw/
```

```python
# Verificar estrutura do dataset
!ls -lh data/raw/
!find data/raw/ -name "*.wav" | head -20
```

---

## 3️⃣ Análise Exploratória

```python
# Carregar módulos do projeto
sys.path.append('.')
from src.data.dataset import SpeakerDataset
from src.utils.helpers import load_config, print_system_info

# Informações do sistema
print_system_info()

# Carregar configuração
config = load_config('config/config.yaml')

# Carregar dataset
dataset = SpeakerDataset('data/raw/wav')
print(f"\n✓ Dataset carregado:")
print(f"  Falantes: {dataset.get_num_speakers()}")
print(f"  Arquivos: {len(dataset.audio_files)}")
```

```python
# Visualizar distribuição de falantes
import matplotlib.pyplot as plt
import seaborn as sns

# Criar DataFrame
data = []
for audio_file, label in zip(dataset.audio_files, dataset.labels):
    speaker_name = dataset.get_speaker_name(label)
    data.append({'speaker_id': speaker_name, 'label': label})

df = pd.DataFrame(data)
speaker_counts = df['speaker_id'].value_counts()

# Plotar distribuição
plt.figure(figsize=(15, 5))
speaker_counts.head(20).plot(kind='bar', color='steelblue', alpha=0.7)
plt.xlabel('ID do Falante')
plt.ylabel('Número de Amostras')
plt.title('Distribuição dos 20 Principais Falantes')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

print(f"\nEstatísticas:")
print(f"  Média: {speaker_counts.mean():.1f}")
print(f"  Desvio: {speaker_counts.std():.1f}")
print(f"  Mín: {speaker_counts.min()}")
print(f"  Máx: {speaker_counts.max()}")
```

```python
# Visualizar amostras de áudio
import IPython.display as ipd

# Selecionar 3 falantes aleatórios
falantes_exemplo = np.random.choice(dataset.get_num_speakers(), 3, replace=False)

for speaker_id in falantes_exemplo:
    indices = [i for i, l in enumerate(dataset.labels) if l == speaker_id]
    if indices:
        audio_file = dataset.audio_files[indices[0]]
        speaker_name = dataset.get_speaker_name(speaker_id)
        
        print(f"\n{'='*60}")
        print(f"Falante: {speaker_name}")
        print(f"Arquivo: {Path(audio_file).name}")
        print(f"{'='*60}")
        
        # Carregar e exibir áudio
        y, sr = librosa.load(audio_file, sr=16000, duration=5.0)
        
        # Player de áudio
        display(ipd.Audio(y, rate=sr))
        
        # Forma de onda
        plt.figure(figsize=(12, 3))
        librosa.display.waveshow(y, sr=sr)
        plt.title(f'Forma de Onda - {speaker_name}')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.show()
        
        # Espectrograma
        plt.figure(figsize=(12, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Espectrograma - {speaker_name}')
        plt.tight_layout()
        plt.show()
```

---

## 4️⃣ Extração de Features

```python
# Dividir dataset
print("Dividindo dataset...")
train_idx, val_idx, test_idx = dataset.split_dataset(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
)

print(f"✓ Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
```

```python
# Função de extração de features
def extrair_features(arquivo_audio, sr=16000, max_len=100):
    """Extrair MFCCs de um arquivo de áudio"""
    try:
        # Carregar áudio
        y, _ = librosa.load(arquivo_audio, sr=sr, duration=5.0)
        
        # Extrair 40 MFCCs
        mfccs = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=40, 
            n_fft=2048, hop_length=512
        )
        
        # Ajustar para tamanho fixo
        if mfccs.shape[1] < max_len:
            mfccs = np.pad(mfccs, ((0, 0), (0, max_len - mfccs.shape[1])), 
                          mode='constant')
        else:
            mfccs = mfccs[:, :max_len]
        
        return mfccs.T  # (tempo, features)
    
    except Exception as e:
        print(f"Erro: {e}")
        return None

# Extrair features para cada conjunto
def carregar_features(indices, desc="Processando"):
    features = []
    labels = []
    
    for idx in tqdm(indices, desc=desc):
        audio_file, label = dataset[idx]
        feat = extrair_features(audio_file)
        
        if feat is not None:
            features.append(feat)
            labels.append(label)
    
    return np.array(features), np.array(labels)

print("\nExtraindo features...")
print("⚠️  Isso pode levar alguns minutos dependendo do tamanho do dataset\n")

X_train, y_train = carregar_features(train_idx, "Train")
X_val, y_val = carregar_features(val_idx, "Val")
X_test, y_test = carregar_features(test_idx, "Test")

print(f"\n✓ Features extraídas:")
print(f"  Train: {X_train.shape}")
print(f"  Val:   {X_val.shape}")
print(f"  Test:  {X_test.shape}")

# Salvar features (opcional)
!mkdir -p data/processed
np.save('data/processed/X_train.npy', X_train)
np.save('data/processed/y_train.npy', y_train)
np.save('data/processed/X_val.npy', X_val)
np.save('data/processed/y_val.npy', y_val)
np.save('data/processed/X_test.npy', X_test)
np.save('data/processed/y_test.npy', y_test)
print("\n✓ Features salvas em data/processed/")
```

---

## 5️⃣ Treinamento Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("="*80)
print("TREINAMENTO: RANDOM FOREST")
print("="*80)

# Achatar features (RF precisa de features 1D)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

print(f"\nShapes achatadas:")
print(f"  Train: {X_train_flat.shape}")
print(f"  Val:   {X_val_flat.shape}")
print(f"  Test:  {X_test_flat.shape}")

# Treinar modelo
print("\nTreinando Random Forest...")
modelo_rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    verbose=2
)

modelo_rf.fit(X_train_flat, y_train)
print("\n✓ Treinamento concluído!")

# Avaliar
print("\nAvaliando modelo...")
rf_train_acc = accuracy_score(y_train, modelo_rf.predict(X_train_flat))
rf_val_acc = accuracy_score(y_val, modelo_rf.predict(X_val_flat))
rf_test_acc = accuracy_score(y_test, modelo_rf.predict(X_test_flat))

print(f"\n📊 Resultados Random Forest:")
print(f"  Acurácia Treino: {rf_train_acc:.4f} ({rf_train_acc*100:.2f}%)")
print(f"  Acurácia Val:    {rf_val_acc:.4f} ({rf_val_acc*100:.2f}%)")
print(f"  Acurácia Teste:  {rf_test_acc:.4f} ({rf_test_acc*100:.2f}%)")

# Salvar modelo
!mkdir -p models
with open('models/random_forest.pkl', 'wb') as f:
    pickle.dump(modelo_rf, f)
print("\n✓ Modelo salvo em models/random_forest.pkl")
```

```python
# Matriz de confusão Random Forest
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred_rf = modelo_rf.predict(X_test_flat)
cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(12, 10))
sns.heatmap(cm_rf, annot=False, cmap='Blues', cbar=True)
plt.title('Matriz de Confusão - Random Forest')
plt.ylabel('Verdadeiro')
plt.xlabel('Predito')
plt.tight_layout()
plt.show()

# Relatório de classificação
print("\nRelatório de Classificação (Random Forest):")
print("="*60)
print(classification_report(y_test, y_pred_rf))
```

---

## 6️⃣ Treinamento CNN 1D

```python
from tensorflow import keras
from tensorflow.keras import layers

print("="*80)
print("TREINAMENTO: CNN 1D")
print("="*80)

# Construir modelo CNN
print("\nConstruindo arquitetura CNN...")

modelo_cnn = keras.Sequential([
    # Input
    keras.Input(shape=(X_train.shape[1], X_train.shape[2])),
    
    # Bloco Convolucional 1
    layers.Conv1D(64, kernel_size=3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),
    
    # Bloco Convolucional 2
    layers.Conv1D(128, kernel_size=3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),
    
    # Bloco Convolucional 3
    layers.Conv1D(256, kernel_size=3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.5),
    
    # Camadas Densas
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(dataset.get_num_speakers(), activation='softmax')
])

# Compilar
modelo_cnn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📐 Arquitetura do Modelo:")
modelo_cnn.summary()

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'models/cnn_melhor.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Treinar
print("\nTreinando CNN...")
print("⚠️  Isso pode levar vários minutos...\n")

historico = modelo_cnn.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print("\n✓ Treinamento concluído!")
```

```python
# Visualizar curvas de treinamento
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Acurácia
ax1.plot(historico.history['accuracy'], label='Treino', linewidth=2)
ax1.plot(historico.history['val_accuracy'], label='Validação', linewidth=2)
ax1.set_xlabel('Época')
ax1.set_ylabel('Acurácia')
ax1.set_title('Acurácia Durante o Treinamento')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Perda
ax2.plot(historico.history['loss'], label='Treino', linewidth=2)
ax2.plot(historico.history['val_loss'], label='Validação', linewidth=2)
ax2.set_xlabel('Época')
ax2.set_ylabel('Perda')
ax2.set_title('Perda Durante o Treinamento')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

```python
# Avaliar CNN
print("\nAvaliando CNN...")
cnn_train_loss, cnn_train_acc = modelo_cnn.evaluate(X_train, y_train, verbose=0)
cnn_val_loss, cnn_val_acc = modelo_cnn.evaluate(X_val, y_val, verbose=0)
cnn_test_loss, cnn_test_acc = modelo_cnn.evaluate(X_test, y_test, verbose=0)

print(f"\n📊 Resultados CNN 1D:")
print(f"  Acurácia Treino: {cnn_train_acc:.4f} ({cnn_train_acc*100:.2f}%)")
print(f"  Acurácia Val:    {cnn_val_acc:.4f} ({cnn_val_acc*100:.2f}%)")
print(f"  Acurácia Teste:  {cnn_test_acc:.4f} ({cnn_test_acc*100:.2f}%)")

# Salvar modelo
modelo_cnn.save('models/cnn_modelo.h5')
print("\n✓ Modelo salvo em models/cnn_modelo.h5")
```

```python
# Matriz de confusão CNN
y_pred_cnn = np.argmax(modelo_cnn.predict(X_test), axis=1)
cm_cnn = confusion_matrix(y_test, y_pred_cnn)

plt.figure(figsize=(12, 10))
sns.heatmap(cm_cnn, annot=False, cmap='Greens', cbar=True)
plt.title('Matriz de Confusão - CNN 1D')
plt.ylabel('Verdadeiro')
plt.xlabel('Predito')
plt.tight_layout()
plt.show()

# Relatório de classificação
print("\nRelatório de Classificação (CNN):")
print("="*60)
print(classification_report(y_test, y_pred_cnn))
```

---

## 7️⃣ Análise Comparativa

```python
print("="*80)
print("COMPARAÇÃO FINAL: RANDOM FOREST vs CNN 1D")
print("="*80)
print()

# Criar tabela de comparação
comparacao = pd.DataFrame({
    'Modelo': ['Random Forest', 'CNN 1D'],
    'Acurácia Treino': [rf_train_acc, cnn_train_acc],
    'Acurácia Validação': [rf_val_acc, cnn_val_acc],
    'Acurácia Teste': [rf_test_acc, cnn_test_acc]
})

print(comparacao.to_string(index=False))
print()

# Determinar vencedor
if cnn_test_acc > rf_test_acc:
    vencedor = "CNN 1D"
    diferenca = (cnn_test_acc - rf_test_acc) * 100
else:
    vencedor = "Random Forest"
    diferenca = (rf_test_acc - cnn_test_acc) * 100

print(f"🏆 VENCEDOR: {vencedor}")
print(f"   Diferença: {diferenca:.2f}% pontos percentuais")
```

```python
# Gráfico de barras comparativo
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(3)
largura = 0.35

bars1 = ax.bar(x - largura/2, 
               [rf_train_acc, rf_val_acc, rf_test_acc],
               largura, label='Random Forest', color='steelblue', alpha=0.8)

bars2 = ax.bar(x + largura/2,
               [cnn_train_acc, cnn_val_acc, cnn_test_acc],
               largura, label='CNN 1D', color='coral', alpha=0.8)

ax.set_xlabel('Conjunto de Dados', fontsize=12)
ax.set_ylabel('Acurácia', fontsize=12)
ax.set_title('Comparação de Desempenho: RF vs CNN', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Treino', 'Validação', 'Teste'])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.1])

# Adicionar valores nas barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()
```

---

## 8️⃣ Resultados e Conclusões

```python
print("="*80)
print("RELATÓRIO FINAL")
print("="*80)
print()

print("📊 DESEMPENHO DOS MODELOS")
print("-" * 80)
print(f"\nRandom Forest:")
print(f"  ├─ Acurácia no teste: {rf_test_acc*100:.2f}%")
print(f"  ├─ Overfitting: {(rf_train_acc - rf_test_acc)*100:.2f}% pontos")
print(f"  └─ Tempo de treinamento: Rápido (< 1 min)")

print(f"\nCNN 1D:")
print(f"  ├─ Acurácia no teste: {cnn_test_acc*100:.2f}%")
print(f"  ├─ Overfitting: {(cnn_train_acc - cnn_test_acc)*100:.2f}% pontos")
print(f"  └─ Tempo de treinamento: Moderado (5-15 min)")

print(f"\n🏆 Modelo Vencedor: {vencedor}")
print(f"   Superioridade: {diferenca:.2f}% pontos percentuais")

print("\n💡 CONCLUSÕES")
print("-" * 80)

if vencedor == "CNN 1D":
    print("""
A CNN 1D demonstrou melhor capacidade de:
  ✓ Capturar padrões temporais nas features de áudio
  ✓ Generalizar para dados não vistos
  ✓ Aprender representações hierárquicas
  
Recomendação: Usar CNN 1D para produção
""")
else:
    print("""
O Random Forest demonstrou melhor desempenho devido a:
  ✓ Eficiência com datasets menores
  ✓ Menor propensão a overfitting
  ✓ Treinamento mais rápido
  ✓ Não requer ajuste fino extenso
  
Recomendação: RF para prototipagem rápida, CNN com mais dados
""")

print("\n📁 ARQUIVOS GERADOS")
print("-" * 80)
print("  ├─ models/random_forest.pkl")
print("  ├─ models/cnn_modelo.h5")
print("  ├─ models/cnn_melhor.h5")
print("  └─ data/processed/*.npy")

print("\n✅ PIPELINE COMPLETO EXECUTADO COM SUCESSO!")
print("="*80)
```

```python
# Salvar resultados finais
resultados_finais = {
    'random_forest': {
        'train_acc': float(rf_train_acc),
        'val_acc': float(rf_val_acc),
        'test_acc': float(rf_test_acc)
    },
    'cnn': {
        'train_acc': float(cnn_train_acc),
        'val_acc': float(cnn_val_acc),
        'test_acc': float(cnn_test_acc),
        'historico': historico.history
    },
    'vencedor': vencedor,
    'diferenca_percentual': float(diferenca),
    'num_falantes': dataset.get_num_speakers(),
    'num_amostras': len(dataset)
}

# Salvar
with open('results/resultados_finais.pkl', 'wb') as f:
    pickle.dump(resultados_finais, f)

print("✓ Resultados salvos em results/resultados_finais.pkl")

# Download dos resultados
from google.colab import files
files.download('results/resultados_finais.pkl')
files.download('models/random_forest.pkl')
files.download('models/cnn_modelo.h5')
```

---

## 📝 Notas Finais

- **Tempo estimado**: 30-60 minutos (depende do tamanho do dataset)
- **Recursos**: Recomendado usar GPU no Colab para CNN
- **Memória**: Mínimo 12GB RAM recomendado
- **Dataset**: VoxCeleb1 completo ~38GB, usar apenas teste para testes rápidos

---

**Autor**: Projeto BCC406 - Redes Neurais  
**Data**: Dezembro 2025  
**Versão**: 1.0
