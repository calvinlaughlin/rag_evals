#!/bin/bash

# Move to script directory first
cd "$(dirname "$0")"  # Move to script directory

# Install required packages from parent directory
pip install -r ../requirements.txt

# Run the experiment on the sample document
python main.py ../sample_document.txt

# Display results
echo "Experiment completed. Results saved to results.json"
