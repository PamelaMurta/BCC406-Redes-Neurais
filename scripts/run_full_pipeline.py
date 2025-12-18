"""
Pipeline Simplificado - Extração de Features e Treinamento
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pickle
import librosa
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras

from src.data.dataset import SpeakerDataset
from src.utils.helpers import load_config, set_random_seeds

print("=" * 80)
print("PIPELINE COMPLETO: RF vs CNN para Identificação de Falantes")
print("=" * 80)
print()

# Configuração
config = load_config('config/config.yaml')
set_random_seeds(config['seeds']['numpy'])
tf.random.set_seed(config['seeds']['tensorflow'])

# Carregar dataset
print("1. Carregando dataset...")
data_dir = config['dataset']['data_dir']
dataset = SpeakerDataset(data_dir)
print(f"   ✓ {len(dataset)} arquivos de {dataset.get_num_speakers()} falantes\n")

# Dividir dataset
print("2. Dividindo dataset...")
train_idx, val_idx, test_idx = dataset.split_dataset(
    train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
    seed=config['seeds']['numpy']
)
print(f"   ✓ Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}\n")

# Função para extrair features
def extract_features(audio_file, sr=16000, max_len=100):
    """Extrair MFCCs de um arquivo de áudio"""
    try:
        y, _ = librosa.load(audio_file, sr=sr, duration=5.0)
        
        # Extrair 40 MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
        
        # Pad ou truncar para tamanho fixo
        if mfccs.shape[1] < max_len:
            mfccs = np.pad(mfccs, ((0, 0), (0, max_len - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :max_len]
        
        return mfccs.T  # (time_steps, n_features)
    except Exception as e:
        print(f"Erro em {audio_file}: {e}")
        return None

# Extrair features
print("3. Extraindo features...")

def load_features(indices, desc):
    features = []
    labels = []
    for idx in tqdm(indices, desc=f"   {desc}"):
        audio_file, label = dataset[idx]
        feat = extract_features(audio_file)
        if feat is not None:
            features.append(feat)
            labels.append(label)
    return np.array(features), np.array(labels)

X_train, y_train = load_features(train_idx, "Train")
X_val, y_val = load_features(val_idx, "Val  ")
X_test, y_test = load_features(test_idx, "Test ")

print(f"\n   Shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}\n")

# ========== RANDOM FOREST ==========
print("=" * 80)
print("TREINAMENTO: RANDOM FOREST")
print("=" * 80)
print()

# Achatar features para RF
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

print("4. Treinando Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=config['seeds']['sklearn'],
    n_jobs=-1,
    verbose=1
)

rf_model.fit(X_train_flat, y_train)
print("   ✓ Treinamento concluído\n")

# Avaliar RF
print("5. Avaliando Random Forest...")
rf_train_acc = accuracy_score(y_train, rf_model.predict(X_train_flat))
rf_val_acc = accuracy_score(y_val, rf_model.predict(X_val_flat))
rf_test_acc = accuracy_score(y_test, rf_model.predict(X_test_flat))

print(f"   Train Accuracy: {rf_train_acc:.4f}")
print(f"   Val Accuracy:   {rf_val_acc:.4f}")
print(f"   Test Accuracy:  {rf_test_acc:.4f}\n")

# Salvar modelo RF
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)
with open(models_dir / 'random_forest.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print(f"   ✓ Modelo salvo em: {models_dir / 'random_forest.pkl'}\n")

# ========== CNN ==========
print("=" * 80)
print("TREINAMENTO: CNN 1D")
print("=" * 80)
print()

print("6. Construindo modelo CNN...")
# Arquitetura CNN 1D
model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    
    # Bloco 1
    keras.layers.Conv1D(64, 3, padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling1D(2),
    keras.layers.Dropout(0.3),
    
    # Bloco 2
    keras.layers.Conv1D(128, 3, padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling1D(2),
    keras.layers.Dropout(0.3),
    
    # Bloco 3
    keras.layers.Conv1D(256, 3, padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dropout(0.5),
    
    # Classificador
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(dataset.get_num_speakers(), activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())
print()

print("7. Treinando CNN...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ],
    verbose=1
)

print("\n   ✓ Treinamento concluído\n")

# Avaliar CNN
print("8. Avaliando CNN...")
cnn_train_loss, cnn_train_acc = model.evaluate(X_train, y_train, verbose=0)
cnn_val_loss, cnn_val_acc = model.evaluate(X_val, y_val, verbose=0)
cnn_test_loss, cnn_test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"   Train Accuracy: {cnn_train_acc:.4f}")
print(f"   Val Accuracy:   {cnn_val_acc:.4f}")
print(f"   Test Accuracy:  {cnn_test_acc:.4f}\n")

# Salvar modelo CNN
model.save(models_dir / 'cnn_model.h5')
print(f"   ✓ Modelo salvo em: {models_dir / 'cnn_model.h5'}\n")

# ========== COMPARAÇÃO FINAL ==========
print("=" * 80)
print("COMPARAÇÃO FINAL: RF vs CNN")
print("=" * 80)
print()

print(f"{'Modelo':<20} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12}")
print("-" * 60)
print(f"{'Random Forest':<20} {rf_train_acc:<12.4f} {rf_val_acc:<12.4f} {rf_test_acc:<12.4f}")
print(f"{'CNN 1D':<20} {cnn_train_acc:<12.4f} {cnn_val_acc:<12.4f} {cnn_test_acc:<12.4f}")
print()

# Determinar vencedor
if cnn_test_acc > rf_test_acc:
    winner = "CNN 1D"
    diff = (cnn_test_acc - rf_test_acc) * 100
else:
    winner = "Random Forest"
    diff = (rf_test_acc - cnn_test_acc) * 100

print(f"🏆 Vencedor: {winner} (diferença de {diff:.2f}%)\n")

# Salvar resultados
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

results = {
    'random_forest': {
        'train_acc': float(rf_train_acc),
        'val_acc': float(rf_val_acc),
        'test_acc': float(rf_test_acc)
    },
    'cnn': {
        'train_acc': float(cnn_train_acc),
        'val_acc': float(cnn_val_acc),
        'test_acc': float(cnn_test_acc)
    },
    'winner': winner,
    'difference': float(diff)
}

with open(results_dir / 'comparison_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"✅ Resultados salvos em: {results_dir / 'comparison_results.pkl'}")
print("\n" + "=" * 80)
print("✅ PIPELINE COMPLETO EXECUTADO COM SUCESSO!")
print("=" * 80)
