"""Test that JSON configs match the original default_model_config."""

from pathlib import Path

from hssm.defaults import default_model_config
from hssm.configs.utils import load_model_config

def compare_likelihood_config(json_likelihood, orig_likelihood, model_name, kind):
    """Compare likelihood configurations."""
    # Check loglik
    if callable(orig_likelihood["loglik"]):
        assert json_likelihood["loglik"] == orig_likelihood["loglik"].__name__, \
            f"{model_name} {kind} loglik mismatch"
    else:
        assert json_likelihood["loglik"] == orig_likelihood["loglik"], \
            f"{model_name} {kind} loglik mismatch"
    
    # Check backend
    assert json_likelihood["backend"] == orig_likelihood["backend"], \
        f"{model_name} {kind} backend mismatch"
    
    # Check bounds if they exist
    if "bounds" in json_likelihood and "bounds" in orig_likelihood:
        orig_bounds = {k: list(v) for k, v in orig_likelihood["bounds"].items()}
        assert json_likelihood["bounds"] == orig_bounds, f"{model_name} {kind} bounds mismatch"
    
    # Check default_priors if they exist
    if "default_priors" in json_likelihood and "default_priors" in orig_likelihood:
        assert json_likelihood["default_priors"] == orig_likelihood["default_priors"], \
            f"{model_name} {kind} default_priors mismatch"
    
    # Check extra_fields
    if "extra_fields" in json_likelihood and "extra_fields" in orig_likelihood:
        assert json_likelihood["extra_fields"] == orig_likelihood["extra_fields"], \
            f"{model_name} {kind} extra_fields mismatch"

def test_model_config(model_name):
    """Test that a model's JSON config matches the original config."""
    print(f"\nTesting {model_name} configuration...")
    
    # Load JSON config
    json_config = load_model_config(model_name)
    
    # Get original config
    original_config = default_model_config[model_name]
    
    # Compare basic fields
    assert json_config["response"] == original_config["response"], \
        f"{model_name} response mismatch"
    assert json_config["list_params"] == original_config["list_params"], \
        f"{model_name} list_params mismatch"
    assert json_config["choices"] == original_config["choices"], \
        f"{model_name} choices mismatch"
    assert json_config["description"] == original_config["description"], \
        f"{model_name} description mismatch"
    
    # Compare likelihoods structure
    assert set(json_config["likelihoods"].keys()) == set(original_config["likelihoods"].keys()), \
        f"{model_name} likelihood keys mismatch"
    
    # Check each likelihood type
    for kind in json_config["likelihoods"]:
        compare_likelihood_config(
            json_config["likelihoods"][kind],
            original_config["likelihoods"][kind],
            model_name,
            kind
        )
    
    print(f"✓ {model_name} configuration matches")

def test_all_configs():
    """Test all model configurations."""
    print("Testing all model configurations...")
    
    # Get all JSON files except split_configs.py and test_configs.py
    config_files = [f.stem for f in Path(__file__).parent.glob("*.json")]
    
    # Test each model configuration
    for model_name in config_files:
        test_model_config(model_name)
    
    print("\n✓ All configurations match!")

if __name__ == "__main__":
    test_all_configs()
