import sys
import subprocess
import os

def run_step(script_name, step_desc):
    print(f"\n{'='*60}")
    print(f"🚀 STEP: {step_desc}")
    print(f"📄 Running: {script_name}")
    print(f"{'='*60}\n")
    
    try:
        # Use full path
        script_path = os.path.join(os.path.dirname(__file__), script_name)
        subprocess.check_call([sys.executable, script_path])
        print(f"\n✅ {step_desc} COMPLETED!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FAILED: {step_desc}")
        print(f"Error Code: {e.returncode}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)

def main():
    print("\n" + "="*80)
    print("🚀 AZIMA v6.3 - FULL TRAINING PIPELINE START")
    print("="*80)
    
    # 1. Data Prep & Labeling
    run_step("prepare_data_complete.py", "Data Preparation & Labeling")
    
    # 2. Base Models (Level 0)
    run_step("train_base_models_v6.py", "Training Base Models (Level 0)")
    
    # 3. Ensemble (Level 1)
    run_step("train_ensemble_v6.py", "Training Ensemble (Level 1)")
    
    # 4. Filter Model (Level 2)
    run_step("train_filter_model.py", "Training Filter Model (Level 2)")
    
    # 5. Backtest Verification
    run_step("backtest_v6.py", "Running Final Backtest")
    
    print("\n" + "="*80)
    print("🎉 FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()
