import os
import pickle

import cv2
import supervision as sv
from ultralytics import YOLOv10, YOLO

from utils.bbox_utils import get_bbox_width, get_center_of_bbox

YOLO_MODEL_PATH = os.getenv('MODEL_PATH', 'models/best.pt')
CONFIDANCE_MODEL = float(os.getenv('CONFIDANCE_MODEL', 0.2))


class Tracker:
    def __init__(self, model_path: str = YOLO_MODEL_PATH):
        self.model = YOLO(model_path,verbose=True)
        self.tracker = sv.ByteTrack()
        self.ball_trace = []

    def detect_frames(self, frames):
        batch_size = 30
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            detections.extend(self.model.predict(batch_frames, conf=CONFIDANCE_MODEL))
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        tracks = {
            "players": [],
            "ball": [],
            "referee": [],
        }
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # convert to supervision detection format

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # convert goalkeeper to player obj

            for obj_index, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[obj_index] = cls_names_inv['player']

            # track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            tracks["players"].append({})
            tracks["referee"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_ids = frame_detection[3]
                track_ids = frame_detection[4]

                if cls_ids == cls_names_inv['player']:
                    tracks["players"][frame_num][track_ids] = {"bbox": bbox}

                elif cls_ids == cls_names_inv['referee']:

                    tracks["referee"][frame_num][track_ids] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_ids = frame_detection[3]
                track_ids = frame_detection[4]

                if cls_ids == cls_names_inv['ball']:
                    tracks["ball"][frame_num][track_ids] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def get_object_tracks_by_detections(self, detections):

        tracks = {
            "players": [],
            "ball": [],
            "referee": [],
        }
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # convert to supervision detection format

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # convert goalkeeper to player obj

            for obj_index, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[obj_index] = cls_names_inv['player']

            # track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            tracks["players"].append({})
            tracks["referee"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_ids = frame_detection[3]
                track_ids = frame_detection[4]

                if cls_ids == cls_names_inv['player']:
                    tracks["players"][frame_num][track_ids] = {"bbox": bbox}

                elif cls_ids == cls_names_inv['referee']:

                    tracks["referee"][frame_num][track_ids] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_ids = frame_detection[3]
                track_ids = frame_detection[4]

                if cls_ids == cls_names_inv['ball']:
                    tracks["ball"][frame_num][track_ids] = {"bbox": bbox}

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        center = (int(x_center), int(y2))
        axes = (int(width), int(0.35 * width))

        cv2.ellipse(
            frame,
            center=center,
            axes=axes,
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_with = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_with // 2
        x2_rect = x_center + rectangle_with // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)
            x1_text = x1_rect + 5
            if track_id > 99:
                x1_text -= 10

            cv2.putText(frame, str(track_id), (int(x1_text), int(y1_rect + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 2)
        return frame

    def draw_ball_trace(self, frame):
        for i in range(1, len(self.ball_trace)):
            start_point = tuple(map(int, self.ball_trace[i - 1]))
            end_point = tuple(map(int, self.ball_trace[i]))
            alpha = i / len(self.ball_trace)
            color = (0, int(255 * alpha), int(255 * (1 - alpha)))
            frame = cv2.line(frame, start_point, end_point, color, 2)
        return frame

    def update_ball_trace(self, ball_dict):
        if ball_dict:
            for _, ball in ball_dict.items():
                ball_position = get_center_of_bbox(ball['bbox'])
                self.ball_trace.append(ball_position)
        # else:
        #     if self.ball_trace:
        #         last_position = np.array(self.ball_trace[-1])
        #
        #         if len(self.ball_trace) > 2:
        #             directions = [np.array(self.ball_trace[i]) - np.array(self.ball_trace[i - 1]) for i in
        #                           range(-1, -min(4, len(self.ball_trace)), -1)]
        #             average_direction = np.mean(directions, axis=0)
        #             predicted_position = last_position + average_direction
        #         elif len(self.ball_trace) > 1:
        #             direction = np.array(self.ball_trace[-1]) - np.array(self.ball_trace[-2])
        #             predicted_position = last_position + direction
        #         else:
        #             predicted_position = last_position
        #
        #         self.ball_trace.append(tuple(predicted_position))
        #     else:
        #         predicted_position = (0, 0)
        #
        # # Limit the trace length to 10
        if len(self.ball_trace) > 10:
            self.ball_trace.pop(0)

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []

        max_frame = len(video_frames)
        max_player_frames = len(tracks.get('players', []))
        max_ball_frames = len(tracks.get('ball', []))
        max_referee_frames = len(tracks.get('referee', []))

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            if frame_num >= max_player_frames or frame_num >= max_ball_frames or frame_num >= max_referee_frames:
                continue

            player_dict = tracks['players'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            referee_dict = tracks['referee'][frame_num]

            # draw players

            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player['bbox'], (0, 0, 255), track_id)

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 255, 255))

            self.update_ball_trace(ball_dict)
            frame = self.draw_ball_trace(frame)

            output_video_frames.append(frame)

        return output_video_frames
