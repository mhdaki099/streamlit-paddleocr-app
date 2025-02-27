#!/bin/bash

# Make directory for PaddleOCR models
mkdir -p ~/.paddleocr/whl

# Create the needed directories for models
mkdir -p inference/det
mkdir -p inference/rec

echo "Setup completed successfully"