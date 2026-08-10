"""
Diagnostic script to check what's wrong with your saved model
Run this to see exactly what attributes are missing
"""

import joblib
import numpy as np

def diagnose_model(model_path='tcgm_credit_scoring_model.pkl'):
    """
    Diagnose issues with saved TCGM model
    """
    print("="*70)
    print("🔍 DIAGNOSING TCGM MODEL")
    print("="*70)
    
    try:
        # Load model
        print(f"\n📂 Loading model from: {model_path}")
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully")
        print(f"   Type: {type(model).__name__}")
        
        # Check all attributes
        print("\n📋 Checking Required Attributes:")
        print("-" * 70)
        
        required_attrs = {
            'init_pred': 'Initial prediction value',
            'base_models': 'List of base estimators',
            'trees': 'Decision trees',
            'learning_rate': 'Learning rate for boosting',
            'n_estimators': 'Number of boosting iterations',
            'max_depth': 'Maximum tree depth',
            'min_samples_leaf': 'Minimum samples per leaf',
            'cost_fp': 'Cost of false positive',
            'cost_fn': 'Cost of false negative'
        }
        
        missing_attrs = []
        present_attrs = []
        
        for attr, description in required_attrs.items():
            if hasattr(model, attr):
                value = getattr(model, attr)
                present_attrs.append(attr)
                if isinstance(value, (list, np.ndarray)):
                    print(f"✅ {attr:20s} : {description} (length: {len(value)})")
                else:
                    print(f"✅ {attr:20s} : {description} = {value}")
            else:
                missing_attrs.append(attr)
                print(f"❌ {attr:20s} : {description} - MISSING!")
        
        # Summary
        print("\n" + "="*70)
        print("📊 DIAGNOSIS SUMMARY")
        print("="*70)
        print(f"Present attributes: {len(present_attrs)}/{len(required_attrs)}")
        print(f"Missing attributes: {len(missing_attrs)}/{len(required_attrs)}")
        
        if missing_attrs:
            print(f"\n❌ Missing: {', '.join(missing_attrs)}")
            print("\n🔧 RECOMMENDED ACTION:")
            print("   Your model is missing critical attributes.")
            print("   You MUST retrain the model with TCGM 0.1.3")
            print("\n   Steps:")
            print("   1. Ensure you have: pip install tcgm==0.1.3")
            print("   2. Use the 'retrain_model_code.py' script provided")
            print("   3. Save the new model")
            print("   4. Replace the old model file")
        else:
            print("\n✅ All required attributes present!")
            print("   Your model should work in Streamlit.")
        
        # Test prediction
        print("\n" + "="*70)
        print("🧪 TESTING PREDICTION")
        print("="*70)
        
        try:
            # Create dummy data (15 features)
            dummy_data = np.random.rand(1, 15)
            print(f"Created test data: shape {dummy_data.shape}")
            
            # Try prediction
            probs = model.predict_proba(dummy_data)
            print(f"✅ Prediction SUCCESSFUL!")
            print(f"   Output shape: {probs.shape}")
            print(f"   Probability: {probs[0, 1]:.4f}")
            
            return True
            
        except AttributeError as e:
            print(f"❌ Prediction FAILED: {e}")
            print("\n   This confirms the model needs to be retrained.")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
        
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        print("   Make sure the file exists in the current directory.")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def check_tcgm_version():
    """Check installed TCGM version"""
    print("\n" + "="*70)
    print("📦 CHECKING TCGM VERSION")
    print("="*70)
    
    try:
        import tcgm
        version = getattr(tcgm, '__version__', 'Unknown')
        print(f"Installed TCGM version: {version}")
        
        if version != '0.1.3':
            print(f"⚠️ WARNING: You have version {version}")
            print(f"   Recommended version: 0.1.3")
            print(f"\n   Update with: pip install tcgm==0.1.3")
        else:
            print("✅ Correct version installed!")
        
    except ImportError:
        print("❌ TCGM not installed!")
        print("   Install with: pip install tcgm==0.1.3")

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║              TCGM Model Diagnostic Tool                       ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Check TCGM version
    check_tcgm_version()
    
    # Diagnose model
    success = diagnose_model()
    
    # Final recommendation
    print("\n" + "="*70)
    print("💡 FINAL RECOMMENDATION")
    print("="*70)
    
    if not success:
        print("""
❌ Your model has compatibility issues and cannot be fixed automatically.

🔧 SOLUTION: You must retrain the model

📝 Steps:
1. Open your training notebook
2. Run this command: pip install tcgm==0.1.3
3. Copy and run the code from 'retrain_model_code.py'
4. This will create new model files
5. Copy the new files to your Streamlit directory
6. Run streamlit app again

⏱️ This should take about 5-10 minutes.
        """)
    else:
        print("""
✅ Your model passed all tests!

If you're still getting errors in Streamlit:
1. Make sure you're using the updated app.py
2. Run: python fix_model.py
3. Restart Streamlit: streamlit run app.py
        """)
    
    print("="*70)