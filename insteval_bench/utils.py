import json

DATASETS = [
    "llmbar_adversarial",
    "llmbar_natural",
    "mtbench",
    "instrusum",
]

def get_dataset_path(dataset_name: str) -> str:
    if dataset_name not in DATASETS:
        message = f"Dataset {dataset_name} not found"
        raise ValueError(message)
    return f"data/{dataset_name}.json"

def open_utf8(file_path, mode="r"):
    """Open a file in UTF-8 encoding"""
    return open(file_path, mode, encoding="utf-8")

def read_json(file_path: str) -> list[dict]:
    """
    Read a JSON/JSONL file and return its contents as a list of dictionaries.
    
    Parameters:
        file_path (str): The path to the JSON file.
        
    Returns:
        list[dict]: The contents of the JSON file as a list of dictionaries.
    """
    try:
        with open_utf8(file_path) as f:
            data = [json.loads(x) for x in f]
        return data
    except json.decoder.JSONDecodeError:
        with open_utf8(file_path) as f:
            data = json.load(f)
        return data


