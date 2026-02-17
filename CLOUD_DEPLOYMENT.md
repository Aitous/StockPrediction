# Cloud Deployment Guide - ML Signal Bundle

This bundle contains everything needed to train TabPFN ML models for trading signal prediction in cloud environments (AWS, GCP, Azure, etc.).

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-cloud.txt
```

### 2. Train Model on TabPFN

```bash
# Use GPU for optimal TabPFN performance
python scripts/train_ml_model.py --model tabpfn

# Or train with LightGBM (faster, no GPU required)
python scripts/train_ml_model.py --model lightgbm

# Auto-detect best available model (tries TabPFN first)
python scripts/train_ml_model.py --model auto
```

### 3. Use Pre-trained Model

```python
from tradingagents.ml.predictor import MLPredictor

# Load the trained model
predictor = MLPredictor.load("data/ml/tabpfn_model.pkl")

# Make predictions
predictions = predictor.predict(feature_data)
```

## Cloud Environment Setup

### AWS EC2 / SageMaker
```bash
# GPU Instance (recommended for TabPFN)
# Use: p3.2xlarge (NVIDIA V100) or g4dn.xlarge (NVIDIA T4)

apt-get update
apt-get install -y python3.10 python3-pip
pip install -r requirements-cloud.txt
```

### Google Cloud (Vertex AI, AI Platform)
```bash
# GPU Instance
# Use: n1-standard-4 with 1x NVIDIA T4 GPU

pip install -r requirements-cloud.txt

# Train via Cloud Functions or Compute Engine
python scripts/train_ml_model.py
```

### Azure ML
```bash
# Create compute cluster in Azure ML
# Use: Standard_NC6 (1x Tesla K80 GPU) or Standard_NC12 (2x Tesla K80)

# Install via Pipeline
pip install -r requirements-cloud.txt
python scripts/train_ml_model.py
```

## Training Parameters

```bash
# Limit training samples (useful for quick testing)
python scripts/train_ml_model.py --max-train-samples 5000

# Full training (recommended for production)
python scripts/train_ml_model.py --model tabpfn

# Build fresh training dataset before training
python scripts/build_ml_dataset.py
python scripts/train_ml_model.py --model tabpfn
```

## Model Output

After training, you'll get:
- `data/ml/tabpfn_model.pkl` - Trained model (pickle format)
- `data/ml/metrics.json` - Performance metrics (accuracy, precision, recall, F1)

## Performance Notes

**TabPFN** (Recommended):
- Requires GPU for optimal performance (~2-5x faster)
- Works with smaller datasets (10K-100K samples ideal)
- Better for tabular financial data
- ~1-5 minutes training on GPU

**LightGBM** (Fallback):
- No GPU required (CPU-friendly)
- Scales to larger datasets (1M+ samples)
- ~5-15 minutes training on CPU
- More memory efficient

## Memory Requirements

- **Minimum**: 4GB RAM
- **Recommended**: 8GB RAM + GPU with 4GB VRAM
- **Training data**: ~200-500MB (parquet format)

## Troubleshooting

### TabPFN Not Installed
```bash
pip install tabpfn
# Or install from source for latest version
pip install git+https://github.com/autogluon/tabtransformer.git
```

### Out of Memory
Reduce training samples:
```bash
python scripts/train_ml_model.py --max-train-samples 2000
```

### CUDA/GPU Not Found
Falls back to LightGBM automatically. Or use:
```bash
python scripts/train_ml_model.py --model lightgbm
```

## Integration with Trading Application

Once trained, the model is used by:
- `tradingagents/dataflows/discovery/scanners/ml_signal.py` - Real-time signal generation
- Web UI displays ML-based "ML Signal" rankings

## Retraining Schedule

Recommended retraining frequency:
- **Daily**: For real-time trading
- **Weekly**: For backtesting
- **Monthly**: For model validation and performance review

For automated retraining in cloud, use:
- AWS: CloudWatch Events + Lambda
- GCP: Cloud Scheduler + Cloud Functions
- Azure: Logic Apps + Azure Functions
