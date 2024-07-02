#!/bin/bash

pip install --no-input \
    numpy==1.26.3 \
    pandas==2.2.0 \
    matplotlib==3.8.2 \
    scipy==1.12.0 \
    pynndescent==0.5.11 \
    scikit-learn==1.4.0 \
    scanpy==1.9.8
pip install --no-input git+https://github.com/saeyslab/FlowSOM_Python.git
pip install --no-input --use-pep517 git+https://github.com/saeyslab/ViVAE.git