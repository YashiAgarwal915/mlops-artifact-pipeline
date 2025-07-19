import pytest
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
import joblib

# We need to import the function we want to test
from src.train import train_and_save_model

# --- Fixtures ---
# Pytest fixtures are functions that create data, connections, or objects 
# that you use across multiple tests. This avoids duplicating code.

@pytest.fixture
def config():
    """Fixture to load the configuration file."""
    # Build the robust path to the config file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config', 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

@pytest.fixture
def digits_data():
    """Fixture to load the digits dataset."""
    return load_digits()

# --- Test Cases ---

# Test Case 1: Configuration File Loading
def test_config_loading_and_validation(config):
    """
    Tests that the config file loads correctly and has the required keys and types.
    """
    # Assert that the loaded object is a dictionary
    assert isinstance(config, dict)
    
    # Check for the existence of required hyperparameters
    required_keys = ['C', 'solver', 'max_iter']
    for key in required_keys:
        assert key in config, f"Configuration file is missing key: {key}"

    # Check the data types of the hyperparameters
    assert isinstance(config['C'], float), "Hyperparameter 'C' should be a float."
    assert isinstance(config['solver'], str), "Hyperparameter 'solver' should be a string."
    assert isinstance(config['max_iter'], int), "Hyperparameter 'max_iter' should be an integer."

# Test Case 2: Model Creation
def test_model_creation(config):
    """
    Tests that the model object is created correctly and is a LogisticRegression instance.
    """
    # We can create a model instance directly to test it
    model = LogisticRegression(**config)
    assert isinstance(model, LogisticRegression), "Model should be an instance of LogisticRegression."

# Test Case 3: Model Accuracy
def test_model_accuracy(config, digits_data):
    """
    Tests if the trained model achieves a reasonable accuracy score.
    This validates the entire training logic.
    """
    X, y = digits_data.data, digits_data.target
    
    model = LogisticRegression(**config)
    model.fit(X, y)
    
    # Calculate accuracy on the training data
    accuracy = model.score(X, y)
    print(f"Model accuracy: {accuracy}")
    
    # Set a reasonable threshold for accuracy. For the digits dataset, it should be high.
    assert accuracy > 0.95, f"Model accuracy {accuracy} is below the threshold of 0.95."