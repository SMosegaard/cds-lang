# Activate the environment
source ./env/bin/activate

# Run the LR classifier with argparse arguments
python src/query_expansion.py "$@"

# Close the virtual envoriment
deactivate