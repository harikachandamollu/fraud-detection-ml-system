import pandas as pd
from drift_detection import ks_drift, population_stability_index

df_ref = pd.read_csv("data/raw/train_transaction.csv").sample(5000, random_state=1)
df_cur = pd.read_csv("data/raw/train_transaction.csv").sample(5000, random_state=2)

feature = "TransactionAmt"

stat, p, drift = ks_drift(df_ref[feature], df_cur[feature])
psi = population_stability_index(df_ref[feature], df_cur[feature])

print("KS drift:", drift)
print("PSI:", psi)