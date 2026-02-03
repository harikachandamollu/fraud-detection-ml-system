import mlflow.sklearn

def load_latest_model():
    # Uses the most recent run in the experiment
    model = mlflow.sklearn.load_model("runs:/f51620accdb54f11a7964f0fd8e5d535/model")
    return model