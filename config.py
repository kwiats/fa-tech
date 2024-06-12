import os

IS_METADATA_OUTPUT = os.getenv("IS_METADATA_OUTPUT", "y").lower() == "n"
IS_METADATA_DISPLAY_ENABLED = (
    os.getenv("IS_METADATA_DISPLAY_ENABLED", "n").lower() == "n"
)
IS_METADATA_SAVE_ENABLED = os.getenv("IS_METADATA_SAVE_ENABLED", "n").lower() == "n"
METADATA_OUTPUT_FILE = os.getenv("METADATA_OUTPUT_FILE", "metadata.txt")
TORCH_DEVICE = os.getenv("TORCH_DEVICE")
MAX_RECORD_FPS = int(os.getenv("MAX_RECORD_FPS", 30))
ADJUSTMENT_PARAMETER = float(os.getenv("ADJUSTMENT_PARAMETER", 0.0025))
QUEUE_MAX_SIZE = int(os.getenv("QUEUE_MAX_SIZE", 100))
YOLO_MODEL_PATH = os.getenv("MODEL_PATH", "models/best_old_model_yolo10.pt")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "output.mp4")
SAVE_FILE_OUTPUT = os.getenv("SAVE_FILE_OUTPUT", "y").lower() == "y"
DISPLAY_SCREEN = os.getenv("DISPLAY_SCREEN", "n").lower() == "y"
LIST_WINDOWS = os.getenv("LIST_WINDOWS", "n").lower() == "y"
APP_NAME = os.getenv("APP_NAME")
VIDEO_FORMAT = os.getenv("VIDEO_FORMAT", "H264")
DETECTED_SCREEN_NAME = os.getenv("DETECTED_SCREEN_NAME", "Detected Frame")
YOLO_MODEL_LOGS = os.getenv("YOLO_MODEL_LOGS", "n").lower() == "y"
