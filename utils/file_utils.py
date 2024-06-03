import time
from typing import Optional

import cv2
import numpy as np


class FileUtils:
    @staticmethod
    def save_output_to_file(frame: np.ndarray, filename: Optional[str] = None) -> str:
        if filename is None:
            filename = FileUtils.generate_filename()
        cv2.imwrite(filename, frame)
        return filename

    @staticmethod
    def generate_filename(prefix: Optional[str] = 'frame_', extension: Optional[str] = 'jpg') -> str:
        current_timestamp = int(time.time())
        return f"{prefix}{current_timestamp}.{extension}"
