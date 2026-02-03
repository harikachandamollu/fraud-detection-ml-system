import mlflow.sklearn

def load_latest_model():
    # Uses the most recent run in the experiment
    model = mlflow.sklearn.load_model("runs:/<RUN_ID>/model")
    return model