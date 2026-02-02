Target column: isFraud

Primary metric:
- ROC-AUC
- Recall at fixed threshold

Validation strategy:
- Stratified train/validation split

Final model:
- LightGBM

Production concerns:
- Data drift (KS-test, PSI)
- Automated retraining