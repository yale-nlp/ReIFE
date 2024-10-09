from typing import Callable
import inspect

# Initialize a registry dictionary
method_registry = {}
parser_registry = {}

# Define a decorator for registering functions
def register_method(name: str) -> Callable:
    """
    Decorator for registering evaluation methods.

    Args:
        name (str): The name of the method to register.

    Returns:
        Callable: The decorator function.
    """
    def decorator(func: Callable):
        if name in method_registry:
            raise ValueError(f"A method with the name '{name}' is already registered.")
        # Capture function metadata
        func_metadata = {
            "name": func.__name__,
            "doc": func.__doc__,
            "signature": inspect.signature(func),
            "module": func.__module__,
        }
        method_registry[name] = {"func": func, "metadata": func_metadata}
        return func
    return decorator

# Define a decorator for registering parsers
def register_parser(name: str) -> Callable:
    """
    Decorator for registering parsers.

    Args:
        name (str): The name of the parser to register.

    Returns:
        Callable: The decorator function.
    """
    def decorator(func: Callable):
        if name in parser_registry:
            raise ValueError(f"A parser with the name '{name}' is already registered.")
        func_metadata = {
            "name": func.__name__,
            "doc": func.__doc__,
            "signature": inspect.signature(func),
            "module": func.__module__,
        }
        parser_registry[name] = {"func": func, "metadata": func_metadata}
        return func
    return decorator

# Define a function to get a method from the registry
def get_method(name: str) -> Callable:
    """
    Get an evaluation method from the registry by name.

    Args:
        name (str): The name of the method to get.

    Returns:
        Callable: The evaluation method.
    """
    if name not in method_registry:
        raise ValueError(f"No method with the name '{name}' is registered.")
    return method_registry[name]["func"]

# Define a function to get a parser from the registry
def get_parser(name: str) -> Callable:
    """
    Get a parser from the registry by name.

    Args:
        name (str): The name of the parser to get.

    Returns:
        Callable: The parser function.
    """
    if name not in parser_registry:
        raise ValueError(f"No parser with the name '{name}' is registered.")
    return parser_registry[name]["func"]

# Get method info
def method_info(name: str) -> None:
    """
    Print information about an evaluation method.

    Args:
        name (str): The name of the method to get information about.

    Returns:
        None
    """
    if name not in method_registry:
        raise ValueError(f"No method with the name '{name}' is registered.")
    print(f"Method: {name}")
    print(f"Module: {method_registry[name]['metadata']['module']}")
    print(f"Signature: {method_registry[name]['metadata']['signature']}")
    print(f"Docstring: {method_registry[name]['metadata']['doc']}")
    print(f"Function: {method_registry[name]['metadata']['name']}")

# Get parser info
def parser_info(name: str) -> None:
    """
    Print information about a parser.

    Args:
        name (str): The name of the parser to get information about.

    Returns:
        None
    """
    if name not in parser_registry:
        raise ValueError(f"No parser with the name '{name}' is registered.")
    print(f"Parser: {name}")
    print(f"Module: {parser_registry[name]['metadata']['module']}")
    print(f"Signature: {parser_registry[name]['metadata']['signature']}")
    print(f"Docstring: {parser_registry[name]['metadata']['doc']}")
    print(f"Function: {parser_registry[name]['metadata']['name']}")


# List methods
def list_methods(verbose: bool = True) -> None:
    """
    List all registered evaluation methods.
    """
    for method in method_registry:
        if verbose:
            if verbose:
                method_info(method)
            else:
                print(f"Method: {method}")
        print("-" * 50)


# List parsers
def list_parsers(verbose: bool = True) -> None:
    """
    List all registered parsers.
    """
    for parser in parser_registry:
        if verbose:
            parser_info(parser)
        else:
            print(f"Parser: {parser}")
        print("-" * 50)