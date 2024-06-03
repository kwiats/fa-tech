import threading
import time
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Deque, List

import cv2
from ultralytics import YOLO

from config import METADATA_OUTPUT_FILE, IS_METADATA_SAVE_ENABLED, IS_METADATA_OUTPUT, IS_METADATA_DISPLAY_ENABLED
from recorder.metadata_displayer import MetadataDisplay
from recorder.screen_recorder import ScreenRecorder
from utils.file_utils import FileUtils
from utils.window_utils import get_window_bounds

track_history = defaultdict(lambda: [])
CLASS_COLORS = {
    "ball": (0, 255, 0),
    "goalkeeper": (255, 0, 0),
    "player": (0, 0, 255),
    "referee": (0, 255, 255)
}

def process_frame_batch(frames: List, model):
    processed_frames = []
    for frame in frames:
        frame = cv2.resize(frame, (640, 480))  # Example resolution
        results = model.predict(frame, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy.tolist()[0]
                class_id = int(box.cls.item())
                label = f"{model.names[class_id]} {box.conf.item():.2f}"

                color = CLASS_COLORS.get(model.names[class_id], (255, 255, 255))

                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                frame = cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        processed_frames.append(frame)

    return processed_frames

def process_frames(frame_queue: deque, processed_frame_queue: deque, running_event: threading.Event, model, batch_size=10, delay=10):
    with ThreadPoolExecutor(max_workers=2) as executor:
        while running_event.is_set() or frame_queue:
            if not frame_queue or len(frame_queue) < batch_size:
                time.sleep(0.01)
                continue

            frames = [frame_queue.popleft() for _ in range(batch_size)]
            future = executor.submit(process_frame_batch, frames, model)
            processed_frames = future.result()

            for frame in processed_frames:
                processed_frame_queue.append(frame)

            if len(processed_frame_queue) > 30 * batch_size:
                for _ in range(batch_size):
                    processed_frame_queue.popleft()

def display_processed_frames(processed_frame_queue: Deque, running_event: threading.Event, record_thread, recorder, delay=5):
    start_time = time.time()
    while running_event.is_set() or processed_frame_queue:
        if not processed_frame_queue or (time.time() - start_time < delay):
            time.sleep(0.01)
            continue

        frame = processed_frame_queue.popleft()
        cv2.imshow("Processed Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running_event.clear()
            if record_thread.is_alive():
                recorder.stop_recording()
                record_thread.join()
            break
    cv2.destroyAllWindows()

def start_recording_and_processing(frame_queue, processed_frame_queue, running_event, filename, model, monitor_region):
    recorder = ScreenRecorder()
    record_thread = threading.Thread(target=recorder.record_screen, args=(frame_queue, filename, monitor_region))
    processing_thread = threading.Thread(target=process_frames, args=(frame_queue, processed_frame_queue, running_event, model))

    record_thread.start()
    processing_thread.start()

    return recorder, record_thread, processing_thread

def main():
    model = YOLO("models/best.pt").to('mps')
    # model.export(format="coreml")
    # coreml_model = YOLO("yolov8n.mlpackage")

    frame_queue = deque()
    processed_frame_queue = deque()
    running_event = threading.Event()
    running_event.set()

    region = get_window_bounds("QuickTime Player")
    if not region:
        print("Unable to find the window bounds for QuickTime Player")
        return

    filename = FileUtils.generate_filename(extension='avi')

    recorder, record_thread, processing_thread = start_recording_and_processing(frame_queue, processed_frame_queue, running_event, filename, coreml_model, region)
    # time.sleep(2)
    display_processed_frames(processed_frame_queue, running_event, record_thread, recorder)

    running_event.clear()
    processing_thread.join()

    actual_duration = recorder.end_record_time - recorder.start_record_time
    if IS_METADATA_OUTPUT:
        metadata_output_file = METADATA_OUTPUT_FILE

        metadata = MetadataDisplay.get_metadata(filename)
        additional_data = {"Actual Recording Duration": actual_duration}
        if IS_METADATA_DISPLAY_ENABLED:
            MetadataDisplay.display_metadata(metadata, additional_data)
        if IS_METADATA_SAVE_ENABLED:
            MetadataDisplay.save_metadata_to_file(metadata, metadata_output_file, additional_data)

if __name__ == "__main__":
    main()
