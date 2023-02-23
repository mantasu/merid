import json

def load_json(path: str) -> dict:
    """Loads JSON file

    Simply reads the specified file that ends with ".json" and parses
    its contents to python values to make up a python dictionary.
    
    Args:
        path (str): The path to .json file
    
    Returns:
        dict: A python dictionary
    """
    with open(path, 'r') as f:
        # Load JSON file to py
        py_dict = json.load(f)
    
    return py_dict