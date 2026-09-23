from pathlib import Path

import yaml

# Find the main/root folder of the project based on where the current Python file is located.
# __file__ : means The location of the current Python file.
# Path(__file__).resolve() : means Convert the path into its full/absolute path and resolve things like .. and symbolic links.
# .parent : moves one folder upward.

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


# read your config.yaml file and bring its settings into Python.
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        #  r : read mode
        # encoding="utf-8" : This tells Python how to interpret the characters in the file.
        # as file : This gives the opened file a temporary variable name: file

        return yaml.safe_load(
            file
        )  # It takes the YAML text and converts it into a Python object, usually a dictionary.
        # Give the result back to whoever called the function.
