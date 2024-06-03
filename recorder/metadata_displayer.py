import cv2
import os


class MetadataDisplay:
    @staticmethod
    def load_video_file(video_file: str) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video file {video_file}")
        return cap

    @staticmethod
    def get_frame_count(cap: cv2.VideoCapture) -> int:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @staticmethod
    def get_fps(cap: cv2.VideoCapture) -> float:
        return cap.get(cv2.CAP_PROP_FPS)

    @staticmethod
    def get_duration(frame_count: int, fps: float) -> float:
        return frame_count / fps if fps > 0 else 0

    @staticmethod
    def get_file_size(video_file: str) -> int:
        return os.path.getsize(video_file)

    @staticmethod
    def get_bitrate(file_size: int, duration: float) -> float:
        return (file_size * 8) / duration if duration > 0 else 0

    @staticmethod
    def get_metadata(video_file: str) -> dict:
        cap = MetadataDisplay.load_video_file(video_file)

        frame_count = MetadataDisplay.get_frame_count(cap)
        fps = MetadataDisplay.get_fps(cap)
        duration = MetadataDisplay.get_duration(frame_count, fps)
        file_size = MetadataDisplay.get_file_size(video_file)
        bitrate = MetadataDisplay.get_bitrate(file_size, duration)

        cap.release()

        return {
            "video_file": video_file,
            "frame_count": frame_count,
            "fps": fps,
            "duration": duration,
            "bitrate": bitrate,
            "file_size": file_size
        }

    @staticmethod
    def display_metadata(metadata: dict, additional_data: dict = None) -> None:
        combined_metadata = metadata.copy()
        if additional_data:
            combined_metadata.update(additional_data)

        print(f"Video File: {combined_metadata['video_file']}")
        print(f"Recording Duration: {combined_metadata['duration']:.2f} seconds")
        print(f"Total Frames: {combined_metadata['frame_count']}")
        print(f"Frames per Second: {combined_metadata['fps']:.2f}")
        print(f"File Size: {combined_metadata['file_size'] / (1024 * 1024):.2f} MB")
        print(f"Bitrate: {combined_metadata['bitrate'] / 1000:.2f} kbps")

        for key, value in combined_metadata.items():
            if key not in {"video_file", "frame_count", "fps", "duration", "file_size", "bitrate"}:
                print(f"{key}: {value}")

    @staticmethod
    def save_metadata_to_file(metadata: dict, output_file: str, additional_data: dict = None) -> None:
        combined_metadata = metadata.copy()
        if additional_data:
            combined_metadata.update(additional_data)

        with open(output_file, 'a') as f:
            f.write(f"Video File: {combined_metadata['video_file']}\n")
            f.write(f"Recording Duration: {combined_metadata['duration']:.2f} seconds\n")
            f.write(f"Total Frames: {combined_metadata['frame_count']}\n")
            f.write(f"Frames per Second: {combined_metadata['fps']:.2f}\n")
            f.write(f"File Size: {combined_metadata['file_size'] / (1024 * 1024):.2f} MB\n")
            f.write(f"Bitrate: {combined_metadata['bitrate'] / 1000:.2f} kbps\n")

            for key, value in combined_metadata.items():
                if key not in {"video_file", "frame_count", "fps", "duration", "file_size", "bitrate"}:
                    f.write(f"{key}: {value}\n")
            f.write("\n")