import mlflow
from codecarbon import OfflineEmissionsTracker

def hello_mlflow():
    # Get the version of MLflow.
    print("MLflow Version:", mlflow.version.VERSION)
    
    # Get the URI of the tracking server.
    print("Tracking URI:", mlflow.tracking.get_tracking_uri())
    
    # Set experiment name
    mlflow.set_experiment("hello_world")
    
    # Start a new MLflow run 
    with mlflow.start_run() as run:
        ''' With MLflow, you can log parameters, metrics, plots, models, artifact, and more.'''
        # mlflow.models.log_model(model=MLModel, artifact_path="path")
        # mlflow.log_figure(fig_plt,"figureName")
        mlflow.log_param("paramName", 5)
        mlflow.log_metric("metricName", 0.95)
        mlflow.log_artifact("helloword.txt")

if __name__ == "__main__":
    tracker = OfflineEmissionsTracker(country_iso_code="CAN")
    tracker.start()
    hello_mlflow()
    tracker.stop()