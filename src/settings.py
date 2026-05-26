import os

BASE_DIR = os.environ.get("CSB_BASE_DIR", "")
DATA_DIR = os.environ.get("CSB_DATA_DIR", "data/")
REMOTE_PATH_TO_PYTHON = os.environ.get("CSB_PYTHON", "venv/bin/python")
REMOTE_BASE_DIR = os.environ.get("CSB_REMOTE_BASE_DIR", "example_code_oracle/")
