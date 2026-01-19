# MLflow-AutoGluon

MLflow plugin for AutoGluon models, enabling seamless model tracking, versioning, and deployment.

## Features

- Log and load AutoGluon `TabularPredictor` models with MLflow
- PyFunc support for standardized inference and deployment
- Model Registry integration
- REST API serving support
- Extensible design for future AutoGluon model types

## Installation

```bash
pip install mlflow-autogluon[tabular]
```

For development:

```bash
pip install mlflow-autogluon[dev,tabular]
```

## Quickstart

```python
import mlflow
import mlflow_autogluon
from autogluon.tabular import TabularPredictor

# Train model
predictor = TabularPredictor(label="target").fit(train_data)

# Log to MLflow
with mlflow.start_run():
    model_info = mlflow_autogluon.log_model(
        autogluon_model=predictor,
        artifact_path="model",
        model_type="tabular",
    )

# Load and predict
loaded_model = mlflow_autogluon.load_model(model_info.model_uri)
predictions = loaded_model.predict(test_data)
```

## PyFunc Usage

Load as PyFunc for deployment:

```python
import mlflow

pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)

# Standard predict
predictions = pyfunc_model.predict(test_data)

# Predict with probabilities
proba = pyfunc_model.predict(
    test_data,
    params={"predict_method": "predict_proba"},
)
```

## API Reference

### `save_model(autogluon_model, path, ...)`

Save an AutoGluon model to a path.

- `autogluon_model`: AutoGluon model instance
- `path`: Local path where model is saved
- `model_type`: Type of model ('tabular', 'multimodal', 'vision', 'timeseries')
- `conda_env`: Conda environment dict or path
- `pip_requirements`: Override default pip requirements

### `log_model(autogluon_model, artifact_path, ...)`

Log an AutoGluon model as an MLflow artifact.

- `artifact_path`: Artifact path relative to run's artifact root
- `registered_model_name`: Name to register in Model Registry

### `load_model(model_uri)`

Load an AutoGluon model from MLflow.

- `model_uri`: URI pointing to the model (e.g., 'runs:/<run_id>/model')
