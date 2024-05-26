# Activate the virtual envoriment
source ./env/bin/activate 

# Run the vectorizer
python src/vectorizer.py

# Run the LR classifier with argparse arguments
python src/LR_classifier.py "$@"

# Run the NN classifier with argparse arguments
python src/NN_classifier.py "$@"

# Close the virtual envoriment
deactivate