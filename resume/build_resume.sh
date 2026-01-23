#!/bin/bash

# Check if rendercv is installed
if ! command -v rendercv &> /dev/null 
then
    echo "Error: rendercv is not installed."
    echo "Please install it via pip: pip install \"rendercv[full]\""
    exit 1
fi

# Run RenderCV
echo "Starting resume build..."
rendercv render Eduardo_Marinho_Resume.yaml

# Clear rendercv intermediary files (typst)
rm -r rendercv_output
