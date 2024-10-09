from ..base_rm import BaseRM, BaseRMAPI

# Initialize a registry dictionary
model_registry = {}


# Define a decorator for registering functions
def register_model(name: str) -> BaseRM | BaseRMAPI:
    """
    Decorator for registering a model.

    Args:
        name (str): The name of the model to register.

    Returns:
        BaseRM | BaseRMAPI: The model class.
    """

    def decorator(
        model: BaseRM | BaseRMAPI,
    ) -> BaseRM | BaseRMAPI:
        if name in model_registry:
            raise ValueError(f"A model with the name '{name}' is already registered.")
        # Capture function metadata
        model_metadata = {
            "name": model.__name__,
            "module": model.__module__,
        }
        model_registry[name] = {"model": model, "metadata": model_metadata}
        return model

    return decorator


# Define a function to get a model from the registry
def get_model(name: str) -> BaseRM | BaseRMAPI:
    """
    Get a model from the registry by name.

    Args:
        name (str): The name of the model to get.

    Returns:
        BaseRM | BaseRMAPI: The model
    """
    if name not in model_registry:
        raise ValueError(f"No model with the name '{name}' is registered.")
    return model_registry[name]["model"]


# Get model info
def model_info(name: str) -> None:
    """
    Print information about a model.

    Args:
        name (str): The name of the model to get information about.

    Returns:
        None
    """
    if name not in model_registry:
        raise ValueError(f"No method with the model '{name}' is registered.")
    print(f"Model: {name}")
    print(f"Module: {model_registry[name]['metadata']['module']}")
    print(f"Model name: {model_registry[name]['metadata']['name']}")


# List models
def list_models(verbose: bool = True) -> None:
    """
    List all registered models
    """
    for model in model_registry:
        if verbose:
            model_info(model)
        else:
            print(f"Model: {model}")
        print("-" * 50)
