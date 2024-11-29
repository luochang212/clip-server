apt-get update && apt-get install -y wget

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -f -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda init bash

PATH=/opt/conda/bin:$PATH

conda install -y jupyterlab

nohup jupyter lab \
    --ip=0.0.0.0 \
    --allow-root \
    --notebook-dir=/ \
    --no-browser > /workspace/jupyter_lab.log 2>&1 &
