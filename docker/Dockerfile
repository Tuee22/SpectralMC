FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y nodejs npm build-essential libboost-all-dev libcairo2-dev pkg-config python3 python3-dev python3-pip git curl wget ffmpeg

# update package managers 
RUN python3 -m pip install --upgrade pip setuptools wheel 
RUN python3 -m pip install --ignore-installed blinker
RUN npm cache clean -f
RUN npm install -g n
RUN n stable

# install python packages
WORKDIR /app
#RUN python3 -m pip install pandas edward gym[classic_control] matplotlib numba pymc3 pystan QuantLib scikit-learn scipy scons seaborn statsmodels pydot graphviz dask[complete] cupy-cuda12x jupyterlab==3.6.6 plotly ipywidgets boto3 fabric tables pyarrow s3fs mypy pandas-stubs sympy torch lightning torchvision torchtext torchmetrics torchdata torchaudio moviepy tensorboard protobuf pydantic pulsar-client apache-flink
COPY requirements.txt /app/
RUN python3 -m pip install -r requirements.txt

# jupyterlab extension installation instructions as per https://plot.ly/python/getting-started/
# if the extension build is failing with a runtime error, you may need more ram in your container
RUN export NODE_OPTIONS=--max-old-space-size=4096
RUN jupyter labextension install @ijmbarr/jupyterlab_spellchecker --no-build
RUN jupyter lab build
RUN unset NODE_OPTIONS

EXPOSE 8888
CMD jupyter lab --ip=0.0.0.0 --port=8888 --LabApp.token="" --notebook-dir='/home/jupyter' --no-browser --allow-root
