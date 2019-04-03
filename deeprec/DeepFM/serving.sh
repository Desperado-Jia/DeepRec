#!/usr/bin/env bash

cd /Users/ceciliojia/Desktop/ML-System/DeepRec/deeprec/DeepFM

docker run -p 8501:8501 \
--mount type=bind,source=./export/,target=/models/DeepFM \
-e MODEL_NAME=DeepFM \
-t tensorflow/serving