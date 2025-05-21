#!/bin/bash

# Install required packages
pip install -q nltk scikit-learn numpy

# Run the experiment on the bundled sample document. Adjust the path to test with a PDF instead.
python main.py sample_document.txt

# Display results
echo "Experiment completed. Results saved to results.json"
cat results.json
