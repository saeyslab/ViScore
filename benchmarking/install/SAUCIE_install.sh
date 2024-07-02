#!/bin/bash

pip install --no-input --use-pep517 git+https://github.com/davnovak/SAUCIE.git
pip install --no-input --upgrade-strategy only-if-needed numpy six scikit-learn