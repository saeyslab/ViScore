#!/bin/bash

pip install --no-input ivis==2.0.11
pip install --no-input --upgrade-strategy only-if-needed six typing-extensions packaging requests decorator wheel==0.43
pip install --no-input --upgrade-strategy only-if-needed numpy