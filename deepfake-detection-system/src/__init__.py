cat > src/config.py << 'EOF' # type: ignore
"""
Project configuration file
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
MODELS_DIR = DATA_ROOT / "models"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    'input_size': (224, 224, 3),
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50
}

# Dataset configuration
DATASET_CONFIG = {
    'supported_video_formats': ['.mp4', '.avi', '.mov', '.MOV'],
    'supported_image_formats': ['.jpg', '.jpeg', '.png'],
    'target_face_size': (224, 224),
    'max_frames_per_video': 100
}

print(f"Project root: {PROJECT_ROOT}")
print(f"Data directory: {DATA_ROOT}")