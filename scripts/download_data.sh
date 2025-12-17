#!/bin/bash

# Script to download and setup VoxCeleb1 dataset

echo "=============================================="
echo "VoxCeleb1 Dataset Download and Setup"
echo "=============================================="
echo ""

# Configuration
VOXCELEB_RAW_DIR="data/voxceleb1_raw"
VOXCELEB_PROCESSED_DIR="data/voxceleb1"
NUM_SPEAKERS=10
MIN_SAMPLES=100

echo "Configuration:"
echo "  - Raw data directory: ${VOXCELEB_RAW_DIR}"
echo "  - Processed data directory: ${VOXCELEB_PROCESSED_DIR}"
echo "  - Number of speakers: ${NUM_SPEAKERS}"
echo "  - Minimum samples per speaker: ${MIN_SAMPLES}"
echo ""

# Create directories
mkdir -p "${VOXCELEB_RAW_DIR}"
mkdir -p "${VOXCELEB_PROCESSED_DIR}"
mkdir -p "data/processed"

echo "=============================================="
echo "IMPORTANT: Manual Download Required"
echo "=============================================="
echo ""
echo "VoxCeleb1 requires registration and authentication."
echo ""
echo "Please follow these steps:"
echo ""
echo "1. Visit: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/"
echo "2. Register for access to VoxCeleb1"
echo "3. Download the VoxCeleb1 development set"
echo "4. Extract the files to: ${VOXCELEB_RAW_DIR}"
echo ""
echo "Expected structure after extraction:"
echo "${VOXCELEB_RAW_DIR}/"
echo "  ├── id10001/"
echo "  ├── id10002/"
echo "  └── ..."
echo ""
echo "=============================================="
echo ""

# Check if raw data exists
if [ -d "${VOXCELEB_RAW_DIR}/id10001" ]; then
    echo "✓ VoxCeleb1 raw data found!"
    echo ""
    echo "Setting up dataset..."
    
    # Run Python script to organize dataset
    python -c "
from src.data.download_voxceleb import setup_voxceleb_dataset

dataset = setup_voxceleb_dataset(
    voxceleb_dir='${VOXCELEB_RAW_DIR}',
    output_dir='${VOXCELEB_PROCESSED_DIR}',
    num_speakers=${NUM_SPEAKERS},
    min_samples=${MIN_SAMPLES}
)

print('\nDataset setup complete!')
print(f'Speakers: {len(dataset)}')
print(f'Total samples: {sum(len(files) for files in dataset.values())}')
"
    
    echo ""
    echo "=============================================="
    echo "Setup Complete!"
    echo "=============================================="
    echo ""
    echo "Dataset is ready for feature extraction."
    echo "Next steps:"
    echo "  1. Run feature extraction: jupyter notebook notebooks/02_feature_extraction.ipynb"
    echo "  2. Or use: python scripts/train_rf.py"
    echo ""
    
else
    echo "✗ VoxCeleb1 raw data not found."
    echo ""
    echo "Please download the dataset first following the instructions above."
    echo ""
    echo "After downloading and extracting, run this script again:"
    echo "  bash scripts/download_data.sh"
    echo ""
fi

echo "=============================================="
