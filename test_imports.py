import sys
from pathlib import Path

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")
print(f"sys.path: {sys.path[:3]}")

try:
    from src import config
    print("✓ Successfully imported src.config")
    print(f"  DATA_FILE: {config.DATA_FILE}")
except Exception as e:
    print(f"✗ Failed to import src.config: {e}")

try:
    from src.data.preprocessing import load_and_preprocess_data
    print("✓ Successfully imported src.data.preprocessing")
except Exception as e:
    print(f"✗ Failed to import preprocessing: {e}")
