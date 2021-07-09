#!/bin/bash

echo "python3 src/download_data.py"
python3 src/download_data.py

echo "python3 src/format_data.py"
python3 src/format_data.py

echo "python3 src/train.py"
python3 src/train.py

echo "python3 src/infer.py"
python3 src/infer.py