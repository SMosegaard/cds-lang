# Create a virtual envoriment called env
python -m venv env

# Activate the virtual envoriment
source ./env/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Download the spaCy model
python -m spacy download en_core_web_md

# Close the virtual envoriment
deactivate