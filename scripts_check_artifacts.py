
import os
from pathlib import Path
import json

# Setup Paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
MODELS_DIR = ROOT_DIR / 'models_v6'

REQUIRED_FILES = [
    'final_model.h5',
    'metadata.json',
    'train_metrics.json',
    'training_history.json',
    'feature_columns.json',
    'scaler.pkl'
]

def check_model_artifacts():
    print(f"🔍 Checking Model Artifacts in {MODELS_DIR}...")
    
    if not MODELS_DIR.exists():
        print(f"❌ Models directory not found: {MODELS_DIR}")
        return

    models_found = 0
    issues_found = 0
    
    # Iterate over subdirectories
    for model_dir in MODELS_DIR.iterdir():
        if model_dir.is_dir() and model_dir.name != 'ensemble':
            models_found += 1
            print(f"\n📁 Model: {model_dir.name}")
            
            missing_files = []
            for filename in REQUIRED_FILES:
                file_path = model_dir / filename
                if not file_path.exists():
                    missing_files.append(filename)
            
            if missing_files:
                print(f"   ⚠️ MISSING: {', '.join(missing_files)}")
                issues_found += 1
            else:
                print("   ✅ All artifacts present.")
                
            # Check Metadata content if present
            meta_path = model_dir / 'metadata.json'
            if meta_path.exists():
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    print(f"   ℹ️ Trained: {meta.get('timestamp', 'Unknown')}")
                    print(f"   ℹ️ Git Rev: {meta.get('git_rev', 'N/A')}")
                except Exception as e:
                    print(f"   ⚠️ Bad Metadata: {e}")

    print("\n" + "="*40)
    print(f"SUMMARY: checked {models_found} models.")
    if issues_found == 0:
        print("✅ SYSTEM INTEGRITY: 100% - All artifacts verified.")
    else:
        print(f"⚠️ SYSTEM INTEGRITY: WARNING - {issues_found} models have missing files.")
    print("="*40)

if __name__ == "__main__":
    check_model_artifacts()
