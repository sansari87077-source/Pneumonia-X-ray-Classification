DL-005 ROC CURVE FIX

Files:
- app_fixed.py: patched Streamlit application.
- evaluate_fixed.py: evaluation script that writes the ROC JSON fields expected by the app.

Installation:
1. Back up your current app.py and evaluate.py.
2. Copy app_fixed.py into the project root and rename it to app.py.
3. Copy evaluate_fixed.py into the same project root and rename it to evaluate.py.
4. Keep the existing models/, reports/, and data/ folders.
5. Make sure data/chest_xray/test/NORMAL and data/chest_xray/test/PNEUMONIA contain test images.
6. Activate your project virtual environment.
7. Run: python evaluate.py
8. Check: Test-Path .\reports\roc_curve.json
9. Restart Streamlit: streamlit run app.py

Important:
- The evaluator uses the same model candidate order as app.py: pneumonia_custom_cnn.keras first, then pneumonia_mobilenetv2.keras.
- It mirrors the preprocessing logic currently in the uploaded app.py. However, the original train.py was not provided, so the evaluation preprocessing still needs to be checked against the exact training pipeline before treating the ROC value as final.
- The files were syntax-checked, but could not be run against your actual model and test images because those project assets were not included with the uploads.
- This is an educational demonstration, not a medical diagnostic tool.
