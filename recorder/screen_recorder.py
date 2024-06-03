import time
from typing import Union, List, Dict
from collections import deque

import cv2
import mss
import numpy as np

from config import MAX_RECORD_FPS, QUEUE_MAX_SIZE, VIDEO_FORMAT, ADJUSTMENT_PARAMETER


class ScreenRecorder:
    def __init__(self):
        self.running: bool = False
        self.frame_queue: deque = deque(maxlen=QUEUE_MAX_SIZE)
        self.frames: int = 0
        self.start_record_time = None
        self.end_record_time = None

    @staticmethod
    def setup_monitor(region:  tuple[int, int, int, int]) -> Dict[str, int]:
        return {
            "top": int(region[1]),
            "left": int(region[0]),
            "width": int(region[2]),
            "height": int(region[3])
        }

    @staticmethod
    def setup_video_writer(monitor: Dict[str, int],
                           output_file: str,
                           video_format: str = 'H264') -> cv2.VideoWriter:
        fourcc: int = cv2.VideoWriter_fourcc(*video_format)
        video_writer = cv2.VideoWriter(output_file, fourcc, MAX_RECORD_FPS, (monitor["width"], monitor["height"]))
        return video_writer

    @staticmethod
    def capture_frame(sct: mss.mss, monitor: Dict[str, int]) -> np.ndarray:
        img: np.ndarray = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return frame

    @staticmethod
    def put_to_storage(frame: np.ndarray, storage: Union[deque, List[np.ndarray]]) -> None:
        if isinstance(storage, deque):
            storage.append(frame)
        elif isinstance(storage, list):
            storage.append(frame)
            if len(storage) > QUEUE_MAX_SIZE:
                storage.pop(0)

    def record_screen(self, storage: Union[deque, List[np.ndarray]],
                      output_file: str,
                      region: tuple[int, int, int, int] = (0, 0, 640, 320)) -> None:
        self.running = True
        monitor: Dict[str, int] = self.setup_monitor(region)
        out: cv2.VideoWriter = self.setup_video_writer(monitor, output_file, VIDEO_FORMAT)

        self.start_record_time = time.perf_counter()
        with mss.mss() as sct:
            while self.running:
                start_time = time.perf_counter()
                frame: np.ndarray = self.capture_frame(sct, monitor)
                self.put_to_storage(frame, storage)
                out.write(frame)
                self.frames += 1
                self._frame_limiter(start_time)
            out.release()
            self.end_record_time = time.perf_counter()

    def stop_recording(self) -> None:
        self.running = False
        print("Recording stopped")

    def _frame_limiter(self, start_time: float) -> None:
        elapsed_time = time.perf_counter() - start_time
        sleep_time = max(1 / MAX_RECORD_FPS - elapsed_time - ADJUSTMENT_PARAMETER, 0)  # Even smaller adjustment
        time.sleep(sleep_time)
