FROM python:3.13-slim

ARG MLFLOW_S3_ENDPOINT_URL
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

ENV MLFLOW_S3_ENDPOINT_URL=${MLFLOW_S3_ENDPOINT_URL}
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

ENV TMPDIR=/tmp
ENV GIT_PYTHON_REFRESH=quiet
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src:/app/src/titanic/training

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip \
	&& pip install --no-cache-dir \
		uv \
		fire==0.7.1 \
		mlflow[extras]==3.8.1 \
		pandas==2.3.3 \
		scikit-learn==1.8.0 \
		setuptools==80.9.0 \
		ydata-profiling==4.18.0

COPY src /app/src