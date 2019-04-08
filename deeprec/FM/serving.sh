#!/usr/bin/env bash

docker run -p 8501:8501 \
--mount type=bind,source=/Users/ceciliojia/Desktop/ML-System/DeepRec/deeprec/FM/export/,target=/models/FM \
-e MODEL_NAME=FM \
-t tensorflow/serving