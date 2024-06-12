import time
import cv2
import threading
from collections import deque, defaultdict
from typing import Deque, Dict, List
import multiprocessing
from ultralytics import YOLO
from config import (
    METADATA_OUTPUT_FILE,
    IS_METADATA_SAVE_ENABLED,
    IS_METADATA_OUTPUT,
    IS_METADATA_DISPLAY_ENABLED,
)
from recorder.metadata_displayer import MetadataDisplay
from recorder.screen_recorder import ScreenRecorder
from utils.file_utils import FileUtils
from utils.window_utils import get_window_bounds

model = YOLO("models/best.pt")
track_history: Dict[str, List] = defaultdict(lambda: [])
CLASS_COLORS = {
    "ball": (0, 255, 0),
    "goalkeeper": (255, 0, 0),
    "player": (0, 0, 255),
    "referee": (0, 255, 255),
}
CLASS_NAMES = ["ball", "goalkeeper", "player", "referee"]

def process_frame(frame_with_index):
    i, frame = frame_with_index
    results = model.predict(frame, conf=0.2, device="mps", verbose=False)
    for result in results:
        boxes = result.boxes.xyxy.tolist()
        scores = result.boxes.conf.tolist()
        classes = result.boxes.cls.tolist()
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            class_idx = int(classes[j])
            class_name = CLASS_NAMES[class_idx]
            label = f"{class_name}: {scores[j]:.2f}"
            color = CLASS_COLORS.get(class_name, (0, 255, 255))
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            frame = cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )
    return i, frame

def process_frames(
        frame_queue: Deque,
        processed_frame_queue: Deque,
        running_event: threading.Event,
        batch_size: int = 10,
):
    with multiprocessing.Pool() as pool:
        while running_event.is_set() or frame_queue:
            frames_to_process = []
            while frame_queue and len(frames_to_process) < batch_size:
                index, frame = frame_queue.popleft()
                frames_to_process.append((index, frame))
            if frames_to_process:
                processed_frames = pool.map(process_frame, frames_to_process)
                processed_frames.sort(key=lambda x: x[0])
                for index, processed_frame in processed_frames:
                    processed_frame_queue.append(processed_frame)
        if frames_to_process:
            processed_frames = pool.map(process_frame, frames_to_process)
            processed_frames.sort(key=lambda x: x[0])
            for index, processed_frame in processed_frames:
                processed_frame_queue.append(processed_frame)

def display_processed_frames(
        processed_frame_queue: deque,
        running_event: threading.Event,
        record_thread,
        recorder,
        target_fps: int = 30
):
    frame_time = 1.0 / target_fps
    frame_count = 0
    total_time = 0
    fps_list = []
    start_time = time.perf_counter()

    while running_event.is_set() or processed_frame_queue:
        if not processed_frame_queue:
            time.sleep(0.001)
            continue

        frame = processed_frame_queue.popleft()

        current_time = time.perf_counter()
        elapsed_time = current_time - start_time

        if elapsed_time < frame_time:
            time.sleep(frame_time - elapsed_time)

        start_time = time.perf_counter()  # Reset start_time after sleeping

        total_time += elapsed_time
        frame_count += 1

        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        if fps < 1000:
            fps_list.append(fps)

        actual_fps = frame_count / total_time if total_time > 0 else 0
        average_fps = sum(fps_list) / len(fps_list) if fps_list else 0

        cv2.putText(frame, f"Actual FPS: {actual_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Average FPS: {average_fps:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Queue Length: {len(processed_frame_queue)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Processed Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            running_event.clear()
            if record_thread.is_alive():
                recorder.stop_recording()
                record_thread.join()
            break

    cv2.destroyAllWindows()

def start_recording_and_processing(
        frame_queue, processed_frame_queue, running_event, filename, model, monitor_region
):
    recorder = ScreenRecorder()
    lock = threading.Lock()
    record_thread = threading.Thread(
        target=recorder.record_screen, args=(frame_queue, filename, monitor_region)
    )
    processing_thread = threading.Thread(
        target=process_frames,
        args=(frame_queue, processed_frame_queue, running_event),
    )
    record_thread.daemon = True
    processing_thread.daemon = True
    record_thread.start()
    processing_thread.start()

    return recorder, record_thread, processing_thread, lock

def main():
    frame_queue = deque()
    processed_frame_queue = deque()
    running_event = threading.Event()
    running_event.set()

    region = get_window_bounds("QuickTime Player")
    if not region:
        print("Unable to find the window bounds for QuickTime Player")
        return

    filename = FileUtils.generate_filename(extension="avi")

    recorder, record_thread, processing_thread, lock = start_recording_and_processing(
        frame_queue, processed_frame_queue, running_event, filename, model, region
    )
    while processed_frame_queue:
        print(f"Queue Length: {len(processed_frame_queue)}")
    display_processed_frames(
        processed_frame_queue, running_event, record_thread, recorder
    )

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
            MetadataDisplay.save_metadata_to_file(
                metadata, metadata_output_file, additional_data
            )

if __name__ == "__main__":
    main()
