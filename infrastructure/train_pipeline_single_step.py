"""
SageMaker Pipeline for training Titanic dataset stored on S3.
Uses SageMaker SDK 3.1.0
"""

import argparse

import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep

# Configuration
INSTANCE_TYPE = "ml.t3.medium"
INSTANCE_COUNT = 1


def create_training_pipeline(
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

    # Create SKLearnProcessor for preprocessing
    sklearn_processor = SKLearnProcessor(
        framework_version="1.4-2",
        role=role_arn,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )

    # Define preprocessing step with file mapping (S3 <-> local)
    preprocessing_step = ProcessingStep(
        name="PreprocessData",
        processor=sklearn_processor,
        code="scripts/preprocess.py",
        # Map S3 inputs to local processing paths so preprocess.py can read files
        inputs=[
            ProcessingInput(
                input_name="input_data",
                source=f"s3://{s3_bucket}/input/",
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="preprocessed_output",
                source="/opt/ml/processing/output",
                destination=f"s3://{s3_bucket}/preprocessed/",
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

    # Create pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[preprocessing_step],
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
        "--pipeline-name",
        type=str,
        default="titanic-training-pipeline",
        help="Name of the pipeline (default: titanic-training-pipeline)",
    )
    args = parser.parse_args()

    # Create the pipeline
    pipeline = create_training_pipeline(
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
