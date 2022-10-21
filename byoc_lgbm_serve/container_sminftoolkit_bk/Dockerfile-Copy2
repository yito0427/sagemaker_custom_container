# Build an image that can do training and inference in SageMaker
# This is a Python 2 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM ubuntu:16.04

MAINTAINER Amazon AI <sage-learner@amazon.com>

ARG CONDA_DIR=/opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        openjdk-8-jdk-headless \
        cmake \
        build-essential \
        gcc \
        g++ \
        git \
        #nginx \
        wget && \
    # python environment
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash Miniconda3-latest-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    export PATH="$CONDA_DIR/bin:$PATH" && \
    conda config --set always_yes yes --set changeps1 no && \
    # lightgbm
    #conda install -q -y numpy scipy scikit-learn pandas flask gevent gunicorn && \
    conda install -q -y numpy scipy scikit-learn pandas gevent && \
    #git clone --recursive --branch stable --depth 1 https://github.com/Microsoft/LightGBM && \
    #cd LightGBM/python-package && python setup.py install && \
    # clean
    apt-get autoremove -y && apt-get clean && \
    conda clean -a -y && \
    rm -rf /usr/local/src/*


    

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
#COPY lightgbm_regression /opt/program


# Upgrading pip and creating symbolic link for python3
ARG PIP=pip3
RUN ${PIP} --no-cache-dir install --upgrade pip
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python
#RUN pip3 install sagemaker-training
RUN ${PIP} install --no-cache --upgrade \
    multi-model-server \
    sagemaker-inference
#    sagemaker-training


###WORKDIR /opt/program
#WORKDIR /opt/ml

# Copy entrypoint script to the image
COPY dockerd-entrypoint.py /usr/local/bin/dockerd-entrypoint.py
RUN chmod +x /usr/local/bin/dockerd-entrypoint.py

#RUN mkdir -p /home/model-server/

# Copy the default custom service file to handle incoming data and inference requests
#COPY model_handler.py /home/model-server/model_handler.py

# Define an entrypoint script for the docker image
ENTRYPOINT ["python", "/usr/local/bin/dockerd-entrypoint.py"]

# Define command to be passed to the entrypoint
###CMD ["serve"]

####################### reference ##########################################

###FROM ubuntu:18.04

# Set a docker label to advertise multi-model support on the container
###LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
###LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true


# Upgrade installed packages
###RUN apt-get update && apt-get upgrade -y && apt-get clean

# Python package management and basic dependencies
###RUN apt-get install -y curl python3.7 python3.7-dev python3.7-distutils

# Register the version in alternatives
###RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

# Set python 3 as the default python
###RUN update-alternatives --set python /usr/bin/python3.7

# Install necessary dependencies for MMS and SageMaker Inference Toolkit
###RUN apt-get -y install --no-install-recommends \
###    build-essential \
###    ca-certificates \
###    openjdk-8-jdk-headless \
###    curl \
###    vim \
###    && rm -rf /var/lib/apt/lists/* \
###    && python --version \
###    && curl -O https://bootstrap.pypa.io/get-pip.py \
###    && python get-pip.py

###RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
###RUN update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3 1

# Install MXNet, MMS, and SageMaker Inference Toolkit to set up MMS
###RUN pip3 --no-cache-dir install mxnet \
###                                multi-model-server \
###                                sagemaker-inference \
###                                retrying

# Copy entrypoint script to the image
###COPY dockerd-entrypoint.py /usr/local/bin/dockerd-entrypoint.py
###RUN chmod +x /usr/local/bin/dockerd-entrypoint.py

###RUN mkdir -p /home/model-server/

# Copy the default custom service file to handle incoming data and inference requests
###COPY model_handler.py /home/model-server/model_handler.py

# Define an entrypoint script for the docker image
###ENTRYPOINT ["python", "/usr/local/bin/dockerd-entrypoint.py"]

# Define command to be passed to the entrypoint
###CMD ["serve"]