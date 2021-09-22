from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


def load_images(dir_path: Path, target_suffix: List[str] = [".jpg"]) -> np.ndarray:
    result = []
    for i in dir_path.iterdir():
        if i.suffix in target_suffix:
            result.append(np.array(Image.open(i)))
    return np.array(result)


def save_image(image: np.ndarray, dir_path: Path, img_name: str) -> None:
    Image.fromarray(image.astype(np.uint8)).save(
        dir_path / img_name
    )
