# prediction_pipeline.py
"""
Weekly Batch Prediction Pipeline
- Fetches latest model version from registry
- Runs batch predictions
"""

from kfp.v2 import dsl
from kfp.v2.dsl import component, Output, Input, Model, Dataset, Metrics
from google_cloud_pipeline_components.v1.batch_predict_job import ModelBatchPredictOp
from typing import NamedTuple


@component(
    base_image="google/cloud-sdk:latest",
    packages_to_install=["google-cloud-aiplatform"]
)
def get_latest_model_version_op(
        project_id: str,
        region: str,
        model_display_name: str,
        model_output: Output[Model]
) -> NamedTuple("Outputs", [("model_resource_name", str), ("model_version", str), ("model_uri", str)]):
    """
    Get the latest/default version of a model from Model Registry
    """
    from google.cloud import aiplatform
    from collections import namedtuple

    aiplatform.init(project=project_id, location=region)

    # Find model by display name
    models = aiplatform.Model.list(
        filter=f'display_name="{model_display_name}"',
        order_by="create_time desc"
    )

    if not models:
        raise ValueError(f"No model found with display_name: {model_display_name}")

    # Get the parent model (latest by creation time)
    parent_model = models[0]

    # Get default version or latest version
    if hasattr(parent_model, 'version_id') and parent_model.version_id:
        model_version = parent_model.version_id
        model_resource_name = parent_model.resource_name
        model_uri = parent_model.uri
    else:
        # For models without versioning, use the model itself
        model_version = "1"
        model_resource_name = parent_model.resource_name
        model_uri = parent_model.uri

    print(f"Selected model: {model_resource_name}")
    print(f"Model version: {model_version}")
    print(f"Model URI: {model_uri}")

    # Set output
    model_output.uri = model_resource_name
    model_output.metadata["model_version"] = model_version
    model_output.metadata["model_display_name"] = model_display_name
    model_output.metadata["artifact_uri"] = model_uri

    output = namedtuple("Outputs", ["model_resource_name", "model_version", "model_uri"])
    return output(model_resource_name, model_version, model_uri)


@component(
    base_image="gcr.io/deeplearning-platform-release/spark:latest",
    packages_to_install=["pyspark==3.3.0"]
)
def prepare_prediction_data_op(
        project_id: str,
        input_data_path: str,
        output_data_path: str,
        prepared_data: Output[Dataset]
) -> NamedTuple("Outputs", [("data_path", str), ("record_count", int)]):
    """
    Prepare data for batch prediction using PySpark
    """
    from mango.data.prediction_processor import PredictionDataProcessor
    from collections import namedtuple

    # Initialize processor from mango package
    processor = PredictionDataProcessor(
        project_id=project_id,
        spark_config={
            "spark.executor.memory": "4g",
            "spark.driver.memory": "4g"
        }
    )

    # Process data
    result = processor.prepare_for_prediction(
        input_path=input_data_path,
        output_path=output_data_path
    )

    # Write metadata
    prepared_data.path = output_data_path
    prepared_data.metadata["record_count"] = result["record_count"]
    prepared_data.metadata["processing_timestamp"] = result["timestamp"]

    output = namedtuple("Outputs", ["data_path", "record_count"])
    return output(output_data_path, result["record_count"])


@component(
    base_image="google/cloud-sdk:latest",
    packages_to_install=["google-cloud-aiplatform", "google-cloud-storage"]
)
def custom_batch_prediction_op(
        project_id: str,
        region: str,
        prediction_image: str,
        machine_type: str,
        model_uri: str,
        model_version: str,
        input_data_path: str,
        output_path: str,
        predictions_output: Output[Dataset],
        metrics_output: Output[Metrics]
) -> NamedTuple("Outputs", [("prediction_path", str), ("prediction_count", int)]):
    """
    Run custom batch prediction job
    Uses mango.prediction.predict entry point
    """
    from google.cloud import aiplatform
    from datetime import datetime
    from collections import namedtuple
    import time

    aiplatform.init(project=project_id, location=region)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_display_name = f"batch-prediction-{timestamp}"

    # Create custom job for prediction
    job = aiplatform.CustomJob(
        display_name=job_display_name,
        worker_pool_specs=[{
            "machine_spec": {
                "machine_type": machine_type,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": prediction_image,
                "command": ["python", "-m", "mango.prediction.predict"],
                "args": [
                    f"--model-uri={model_uri}",
                    f"--model-version={model_version}",
                    f"--input-data-path={input_data_path}",
                    f"--output-path={output_path}",
                    f"--project-id={project_id}",
                ],
            },
        }],
    )

    # Run job
    print(f"Starting batch prediction job: {job_display_name}")
    job.run(sync=True)

    print(f"Job completed: {job.state}")

    # Set outputs
    predictions_output.path = output_path
    predictions_output.metadata["job_name"] = job_display_name
    predictions_output.metadata["model_version"] = model_version

    # Log metrics
    metrics_output.log_metric("prediction_job_completed", 1)
    metrics_output.log_metric("model_version_used", int(model_version) if model_version.isdigit() else 0)

    output = namedtuple("Outputs", ["prediction_path", "prediction_count"])
    return output(output_path, 0)  # Count can be retrieved from actual prediction results


@dsl.pipeline(
    name="mango-prediction-pipeline",
    description="Weekly batch prediction pipeline",
    pipeline_root=None,
)
def prediction_pipeline(
        project_id: str,
        region: str,
        prediction_image: str,
        processing_image: str,
        machine_type: str,
        model_display_name: str,
        input_data_path: str = "gs://bucket/prediction/input",
):
    """
    Batch prediction pipeline workflow
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bucket_name = input_data_path.split("/")[2]

    # Step 1: Get latest model version from registry
    get_model_task = get_latest_model_version_op(
        project_id=project_id,
        region=region,
        model_display_name=model_display_name
    )

    # Step 2: Prepare prediction data
    prepared_data_path = f"gs://{bucket_name}/prediction_prepared/{model_display_name}/{timestamp}"

    prepare_data_task = prepare_prediction_data_op(
        project_id=project_id,
        input_data_path=input_data_path,
        output_data_path=prepared_data_path
    )

    # Step 3: Run batch prediction
    prediction_output_path = f"gs://{bucket_name}/predictions/{model_display_name}/{timestamp}"

    prediction_task = custom_batch_prediction_op(
        project_id=project_id,
        region=region,
        prediction_image=prediction_image,
        machine_type=machine_type,
        model_uri=get_model_task.outputs["model_uri"],
        model_version=get_model_task.outputs["model_version"],
        input_data_path=prepare_data_task.outputs["data_path"],
        output_path=prediction_output_path
    )

    prediction_task.after(prepare_data_task, get_model_task)
