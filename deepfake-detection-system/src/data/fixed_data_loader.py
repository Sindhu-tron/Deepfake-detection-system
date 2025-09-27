"""
Data loader using the fixed splits without data leakage
"""

from .data_loader import DeepfakeDataLoader
import json

class FixedDataLoader(DeepfakeDataLoader):
    """Data loader with fixed splits"""
    
    def __init__(self, config_path="config/fixed_dataset_config.json"):
        super().__init__(config_path)

# Test the fixed loader
if __name__ == "__main__":
    loader = FixedDataLoader()
    info = loader.get_dataset_info()
    
    print("Fixed dataset info:")
    for split, data in info.items():
        if isinstance(data, dict) and 'total' in data:
            print(f"  {split}: {data['total']} images ({data.get('real', 0)} real, {data.get('fake', 0)} fake)")