### Quick Overview:
- Rows: 590,540
- Columns: 434
- float64: 399
- int64: 4
- object: 31 (categorical)
- Memory: ~1.9 GB
- Target distribution: 0 -> 96.5%, 1-> 3.5%

### Summary:
EDA reveals the dataset is extremely high-dimensional, sparse, imbalanced, mixed featured types, with significant missing identity information and high-cardinality categorical features. These characteristics strongly favor tree-based gradient boosting models like LightGBM, XGBoost or CatBoost, as they naturally handle non-linear feature interactions, missing values, categorical variables, class imbalance and require minimal preprocessing compared to linear or neural models.

*Why tree-based models?*

| Logistic Regression | Neural Networks | Gradient-boosted tree models |
|---------------------|-----------------|------------------------------|
| Linear decision boundary | Sparse categorical data | Learns non-linear interactions |
| Cannot model complex interactions like <br> Device x E-mail x Amount | Missing values everywhere | Missing values are signal, not noise |
| Requires heavy feature engineering | Very large dataset -> expensive | Robust to feature scaling |
| | Harder to debug and explain | Strong performance on imbalanced data |
| | Production-proven in fraud detection |

- k-NN : Curse of dimensionality + slow
- kernel SVM: O(N<sup>2</sup>) scaling + memory explosion

*Why Gradient-boosted tree models?*

| Decision trees (baseline) | Random Forest |
|---------------------------|---------------|
| + Fast | + Handles non-linearity |
| + Interpretable | + Reduces overfitting |
| + Good for sanity checks | - Slow on large data |
| - But usually underfit | -Memory heavy |
| | - Not great for extreme imbalanace |

*How Gradient-boosting helps? ( LightGBM / XGBoost / CatBoost )*

| Problem | How it solves |
|---------|---------------|
| High dimensionality | Feature selection via splits |
| Imbalanced data | Class weights / scale_pos_weight |
| Missing values | Native handling |
| Non-linear patterns | Tree depth & interactions |
| Large dataset | Histogram-based Learning | 

*1. <ins>Light GBM</ins> - baseline & iteration (fastest, best debugging, feature imp, hyperparameter exploration)*
- <font color="green">Fast baselines and many useless features</font>
- <font color="green">Quick feature importance feedback</font>
- <font color="red">Needs categorical encoding</font>
- <font color="red">Can overfit rare categories if not regularized</font>

*2. <ins>CatBoost</ins> - final performance (best categorical handling, less leakage risk, often higher PR-AUC in fraud)*
- <font color="green">Learns category risk safely and interactions automatically</font>
- <font color="green">Regularizes rare categories better than LightGBM</font>
- <font color="red">Slower than LightGBM</font>
- <font color="red">Slightly heavier memory</font>
- <font color="red">Less flexible hyperparameter space</font>

*2. <ins>XGBoost</ins> -optional validation(only when robust confirmation is required)*
- <font color="green">Extremely robust and strong regularization</font>
- <font color="green">Excellent for structured numeric data</font>
- <font color="red">Slower on large datasets and higher memory usage</font>
- <font color="red">Less friendly with many sparse features and categorical handling still manual</font>

For this dataset, all three models are viable, but they serve different purposes. LightGBM is the best first choice due to its speed, scalability, and strong performance on high-dimensional sparse data, making it ideal for rapid iteration and feature selection. CatBoost is particularly well-suited here because the data contains many high-cardinality categorical variables and rare categories; its native categorical handling and ordered target encoding reduce leakage and overfitting, often yielding better PR-AUC in fraud problems. XGBoost is robust but generally slower and less memory-efficient for datasets of this size.

#### Metrics:
Since Accuracy becomes misleading, now relying on metrics ROC-AUC, Precision/Recall, F1-score, **PR-AUC** and also tree-based boosting models optimize ranking-based metrics (AUC) very well.

#### Feature Engineering Notes:
- Used frequency encoding for categorical features
- Added missing indicators for high-missing columns
- Dropped constant features
- Filled remaining missing values with 0
- Applied standard scaling

#### Evaluation & Drift Notes
- Default probability threshold is suboptimal
- Threshold chosen based on recall constraint
- KS-test and PSI used for drift detection
- Drift triggers retraining