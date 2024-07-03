#!/bin/bash

pip install --no-input --use-pep517 git+https://github.com/davnovak/SAUCIE.git
pip install --no-input --upgrade-strategy only-if-needed six typing-extensions packaging requests decorator wheel==0.43
pip install --no-input --upgrade-strategy only-if-needed numpy six scikit-learn