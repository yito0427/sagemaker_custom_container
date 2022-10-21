FROM python:3.8
WORKDIR /usr/src/app
RUN apt-get update && apt-get upgrade -y && apt-get install -y openjdk-11-jdk-headless
RUN pip install --no-cache-dir numpy multi-model-server sagemaker-inference
COPY dockerd-entrypoint.py ./
ENTRYPOINT ["python", "/usr/src/app/dockerd-entrypoint.py"]