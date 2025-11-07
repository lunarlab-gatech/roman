from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel

class PathParams(BaseModel):
    path_to_dataset_folder: str
    path_to_robot_folder: str
    dataset_version_number: str
    path_background_img: str
    background_img_x_edge: float

    @classmethod
    def from_dict(cls, params_dict: dict) -> PathParams:
        return cls(**params_dict)
    
    def get_full_path_to_robot_folder(self) -> Path:
        return Path(self.path_to_dataset_folder) / self.dataset_version_number / self.path_to_robot_folder
    
    def get_full_path_to_background_img(self) -> Path:
        return Path(self.path_to_dataset_folder) / self.dataset_version_number / self.path_background_img