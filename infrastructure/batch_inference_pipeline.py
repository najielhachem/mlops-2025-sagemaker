"""
SageMaker Multi-Steps Pipeline for training Titanic dataset stored on S3.
Uses SageMaker SDK 3.1.0
"""

import argparse

import sagemaker
from sagemaker.workflow.pipeline import Pipeline

# Configuration
INSTANCE_COUNT = 1
FRAMEWORK_VERSION = "1.4-2"
MODEL_PACKAGE_GROUP_NAME = (
    "titanic-model-group"  # choose any name; created if not existing
)


import boto3
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep

INSTANCE_COUNT = 1
FRAMEWORK_VERSION = "1.4-2"


def _get_model_data_from_package(model_package_arn: str) -> str:
    """
    Resolve model.tar.gz S3 path from a model package ARN in the registry.
    """
    sm = boto3.client("sagemaker")

    resp = sm.describe_model_package(ModelPackageName=model_package_arn)
    # Most common case: single container
    model_data_s3 = resp["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
    return model_data_s3


def create_batch_inference_pipeline(
    pipeline_name: str,
    s3_bucket: str,
    model_package_arn: str,
    role_arn: str | None = None,
) -> Pipeline:
    """
    Batch inference pipeline:

      1. Preprocess raw inference data (scripts/preprocess.py)
      2. Featurize preprocessed data (scripts/featurize.py)
      3. Run batch inference via a ProcessingStep (scripts/inference.py),
         using the model artifacts from a Model Package in the registry.
    """

    if s3_bucket is None:
        raise ValueError("s3_bucket must be provided")

    session = sagemaker.Session()

    if role_arn is None:
        role_arn = sagemaker.get_execution_role()

    print(f"Using S3 bucket: {s3_bucket}")
    print(f"Using IAM role: {role_arn}")
    print(f"Using model package ARN: {model_package_arn}")

    # ----------------- Resolve model.tar.gz from the registry -----------------
    model_data_s3 = _get_model_data_from_package(model_package_arn)
    print(f"Resolved model artifacts at: {model_data_s3}")

    # ----------------- Shared SKLearnProcessor -----------------
    sklearn_processor = SKLearnProcessor(
        framework_version=FRAMEWORK_VERSION,
        role=role_arn,
        instance_count=INSTANCE_COUNT,
        instance_type="ml.t3.medium",  # adjust if needed
        sagemaker_session=session,
    )

    # ----------------- Step 1: Preprocess -----------------
    preprocess_step = ProcessingStep(
        name="Preprocess",
        processor=sklearn_processor,
        code="scripts/preprocess.py",
        inputs=[
            ProcessingInput(
                input_name="raw_data",
                source=f"{s3_bucket}/input/",  # pipeline param
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="preprocessed",
                source="/opt/ml/processing/output",
            ),
        ],
        job_arguments=[
            "--input-dir",
            "/opt/ml/processing/input/",
            "--output-dir",
            "/opt/ml/processing/output/",
        ],
    )

    preprocessed_s3 = preprocess_step.properties.ProcessingOutputConfig.Outputs[
        "preprocessed"
    ].S3Output.S3Uri

    # ----------------- Step 2: Featurize -----------------
    featurize_step = ProcessingStep(
        name="Featurize",
        processor=sklearn_processor,
        code="scripts/featurize.py",
        inputs=[
            ProcessingInput(
                input_name="preprocessed",
                source=preprocessed_s3,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="featurized",
                source="/opt/ml/processing/output",
            ),
            ProcessingOutput(
                output_name="artifacts",
                source="/opt/ml/processing/artifacts",
            ),
        ],
        job_arguments=[
            "--input-dir",
            "/opt/ml/processing/input",
            "--output-dir",
            "/opt/ml/processing/output",
            "--artifacts-dir",
            "/opt/ml/processing/artifacts",
        ],
    )

    featurized_s3 = featurize_step.properties.ProcessingOutputConfig.Outputs[
        "featurized"
    ].S3Output.S3Uri

    # ----------------- Step 3: BatchInference (ProcessingStep) -----------------
    inference_step = ProcessingStep(
        name="BatchInference",
        processor=sklearn_processor,
        code="scripts/inference.py",
        inputs=[
            # Featurized data
            ProcessingInput(
                input_name="features",
                source=featurized_s3,
                destination="/opt/ml/processing/input",
            ),
            # Model artifacts from the registry
            ProcessingInput(
                input_name="model_artifacts",
                source=model_data_s3,
                destination="/opt/ml/processing/model",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="predictions",
                source="/opt/ml/processing/output",
                destination=f"{s3_bucket}/output/",  # pipeline param
            )
        ],
        job_arguments=[
            "--features-dir",
            "/opt/ml/processing/input",
            "--model-dir",
            "/opt/ml/processing/model",
            "--output-dir",
            "/opt/ml/processing/output",
        ],
    )

    # ----------------- Pipeline definition -----------------
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[preprocess_step, featurize_step, inference_step],
        sagemaker_session=session,
    )

    return pipeline


def main():
    """Execute the pipeline creation and submission."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="SageMaker pipeline for preprocessing Titanic dataset"
    )
    parser.add_argument(
        "--input-bucket",
        type=str,
        help="S3 bucket containing input data",
    )
    parser.add_argument(
        "--model-arn",
        type=str,
        help="Deployed Model Package ARN for inference",
    )
    parser.add_argument(
        "--pipeline-name",
        type=str,
        default="batch-inference-pipeline",
        help="Name of the pipeline (default: batch-inference-pipeline)",
    )
    args = parser.parse_args()

    # Create the pipeline
    pipeline = create_batch_inference_pipeline(
        pipeline_name=args.pipeline_name,
        s3_bucket=args.input_bucket,
        model_package_arn=args.model_arn,
    )

    # Define pipeline
    pipeline_definition = pipeline.definition()
    print("Pipeline definition:")
    print(pipeline_definition)

    # Create or update the pipeline
    pipeline.upsert(role_arn=sagemaker.get_execution_role())

    # Submit the pipeline for execution
    execution = pipeline.start()

    print(f"Pipeline {pipeline.name} submitted for execution")
    print(f"Execution ARN: {execution.arn}")

    # Wait for execution to complete
    execution.wait()

    print(f"Pipeline execution status: {execution.get_status()}")


if __name__ == "__main__":
    main()
