# https://ko-ya346.hatenablog.com/entry/2022/11/21/005003
# https://github.com/Kaggle/docker-python/releases/tag/46be4b76d45141f69af2ebc251b6603c4d3eed30045028e54eabb43c5e9e1795
FROM gcr.io/kaggle-gpu-images/python:v135

RUN python3 -m pip install --upgrade pip \
    &&  pip install --no-cache-dir \
    black \
    isort \
    jupyterlab_code_formatter \
    jupyterlab-git \
    ipywidgets==7.7.5

RUN pip install --no-cache-dir \
    hydra-core slack_sdk

# https://github.com/tensorflow/io/issues/1755#issuecomment-1423908869
RUN python -m pip install tensorflow-io 
RUN python -m pip uninstall -y tensorflow-io

RUN pip install --no-cache-dir \
    faiss-gpu==1.7.2 sentence-transformers blingfire==0.1.8 peft==0.4.0 datasets==2.14.3 trl==0.5.0

