import os
from pathlib import Path

base_path = Path(__file__).parent.parent.parent

def get_project_base_path():
    return base_path.resolve()

def get_resource_abs_path(res_name):
    return (base_path / (f"sprites-models/{res_name}")).resolve()