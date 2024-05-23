# Activate the environment
source ./env/bin/activate

# Run the code
python src/query_expansion.py "$@"

# Close the environment
deactivate