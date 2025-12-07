"""
SageMaker Multi-Steps Pipeline for training Titanic dataset stored on S3.
Uses SageMaker SDK 3.1.0
"""

import argparse

import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

# Configuration
INSTANCE_COUNT = 1
FRAMEWORK_VERSION = "1.4-2"


def create_multi_step_pipeline(
    pipeline_name: str,
    s3_bucket: str,
    role_arn: str = None,
):
    """
    Create a SageMaker pipeline with a single preprocessing step.

    Args:
        pipeline_name: Name of the pipeline
        s3_bucket: S3 bucket for input/output data
        role_arn: IAM role ARN for SageMaker execution

    Returns:
        Pipeline object
    """

    # Initialize SageMaker session
    session = sagemaker.Session()

    # Use provided bucket or default SageMaker bucket
    if s3_bucket is None:
        raise ValueError("s3_bucket must be provided")

    # Use provided role or get default execution role
    if role_arn is None:
        role_arn = sagemaker.get_execution_role()

    print(f"Using S3 bucket: {s3_bucket}")
    print(f"Using IAM role: {role_arn}")

    # Create SKLearnProcessor for preprocessing and feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version=FRAMEWORK_VERSION,
        role=role_arn,
        instance_count=INSTANCE_COUNT,
        instance_type="ml.t3.medium",
    )

    # Define preprocessing step with file mapping (S3 <-> local)
    preprocess_step = ProcessingStep(
        name="Preprocess",
        processor=sklearn_processor,
        code="scripts/preprocess.py",
        # Map S3 inputs to local processing paths so preprocess.py can read files
        inputs=[
            ProcessingInput(
                input_name="raw_data",
                source=f"s3://{s3_bucket}/input/",
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="preprocessed",
                source="/opt/ml/processing/output",
            ),
        ],
        # Pass local mapped paths to the preprocessing script
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

    # ---------- Step 2: featurize ----------
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

    # ---------- Step 3: train ----------
    estimator = SKLearn(
        entry_point="scripts/train.py",
        role=role_arn,
        instance_type="ml.m5.large",
        instance_count=INSTANCE_COUNT,
        framework_version=FRAMEWORK_VERSION,
        sagemaker_session=session,
    )

    train_step = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=featurized_s3,
                content_type="text/csv",
            ),
            "test": TrainingInput(
                s3_data=featurized_s3,
                content_type="text/csv",
            ),
        },
    )

    return Pipeline(
        name=pipeline_name,
        steps=[preprocess_step, featurize_step, train_step],
        sagemaker_session=session,
    )


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
        "--pipeline-name",
        type=str,
        default="titanic-multi-step-pipeline",
        help="Name of the pipeline (default: titanic-multi-step-pipeline)",
    )
    args = parser.parse_args()

    # Create the pipeline
    pipeline = create_multi_step_pipeline(
        pipeline_name=args.pipeline_name,
        s3_bucket=args.input_bucket,
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
