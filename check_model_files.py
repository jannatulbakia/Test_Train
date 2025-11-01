"""
Script to check if model files exist and can be loaded
Run this to verify model files before deployment
"""

import os
import joblib

required_files = [
    'best_model.pkl',
    'scaler.pkl',
    'label_encoders.pkl',
    'feature_columns.pkl'
]

print("=" * 80)
print("MODEL FILES CHECKER")
print("=" * 80)

print(f"\nCurrent directory: {os.getcwd()}")
print(f"\nFiles in current directory:")
for file in os.listdir('.'):
    if file.endswith('.pkl'):
        size = os.path.getsize(file)
        print(f"  ✓ {file} ({size:,} bytes)")

print("\n" + "-" * 80)
print("Checking required files:")
print("-" * 80)

all_exist = True
for file in required_files:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"✓ {file} exists ({size:,} bytes)")
        try:
            loaded = joblib.load(file)
            print(f"  ✓ Can be loaded (type: {type(loaded).__name__})")
        except Exception as e:
            print(f"  ✗ Cannot load: {e}")
            all_exist = False
    else:
        print(f"✗ {file} NOT FOUND")
        all_exist = False

print("\n" + "=" * 80)
if all_exist:
    print("✅ All model files exist and can be loaded!")
    print("✅ Ready for deployment")
else:
    print("❌ Some model files are missing or cannot be loaded")
    print("❌ Please run: python train_model.py")
print("=" * 80)

