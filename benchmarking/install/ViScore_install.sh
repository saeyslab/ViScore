#!/bin/bash

pip install --no-input \
    numpy==1.22.4 \
    scikit-learn==1.3.2 \
    scipy==1.11.4 \
    pynndescent==0.5.11 \
    matplotlib==3.8.2 \
    pyemd==1.0.0 \
    scanpy==1.9.8
pip install --no-input --use-pep517 git+https://github.com/saeyslab/ViScore.git
