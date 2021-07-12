import numpy as np
from pathlib import Path
from PIL import Image
from typing import List


def load_images(dir_path: Path, target_suffix: List[str] = [".jpg"]) -> np.ndarray:
    result = []
    for i in dir_path.iterdir():
        if i.suffix in target_suffix:
            result.append(np.array(Image.open(i)))
    return np.array(result)
