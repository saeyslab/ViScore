#!/bin/bash

pip install --no-input scikit-learn scipy
pip install --use-pep517 git+https://github.com/davnovak/SQuad-MDS.git
pip install --no-input --upgrade-strategy only-if-needed numpy