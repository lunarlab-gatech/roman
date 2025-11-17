from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel
from ..utils import expandvars_recursive

class PathParams(BaseModel):
    path_to_dataset_folder: str
    path_to_robot_folder: str
    dataset_version_number: str
    path_background_img: str
    background_img_x_edge: float

    @classmethod
    def from_dict(cls, params_dict: dict, check_background_img_path: bool = True) -> PathParams:
        path_params_class = cls(**params_dict)

        # Ensure both full paths exist
        if check_background_img_path:
            if not os.path.isfile(expandvars_recursive(str(path_params_class.get_full_path_to_background_img()))):
                raise ValueError("Parameters for background_img do not point to an existing file!")

        return path_params_class
    
    def get_full_path_to_robot_folder(self) -> Path:
        return Path(self.path_to_dataset_folder) / self.dataset_version_number / self.path_to_robot_folder
    
    def get_full_path_to_background_img(self) -> Path:
        return Path(self.path_to_dataset_folder) / self.dataset_version_number / self.path_background_img