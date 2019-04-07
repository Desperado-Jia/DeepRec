#!/usr/bin/env bash

docker run -p 8501:8501 \
--mount type=bind,source=/Users/ceciliojia/Desktop/ML-System/DeepRec/deeprec/DeepFM/export/,target=/models/DeepFM \
-e MODEL_NAME=DeepFM -t tensorflow/serving
