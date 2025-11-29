FROM python:3.11-slim
WORKDIR /app

# install uv
RUN pip install uv

# copy toml
COPY pyproject.toml uv.lock README.md ./

# run uv to install dependencies
RUN uv sync --locked --no-install-project

# RUN uv sync --locked --no-dev --no-editable
COPY src ./src
RUN uv sync --no-dev --no-editable

# Copy scripts and pipelines
COPY scripts/ ./scripts
COPY pipelines/ ./pipelines

CMD ["bash", "pipelines/train_pipeline.sh"]
