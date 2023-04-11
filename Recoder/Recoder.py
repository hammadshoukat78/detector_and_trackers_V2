import cv2
import os
from datetime import datetime


class WindowRecorder:
    def __init__(self, video_duration, output_dir='Recordings'):
        if video_duration < (5 / 60) or video_duration > 59:
            raise ValueError("Video duration must be between 5 seconds and 59 minutes.")

        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.video_duration_seconds = video_duration * 60
        self.frame_rate = 29
        self.frames_received = 0
        self.seconds_recorded = 0

    def record(self, frame):
        self.frames_received += 1

        if self.frames_received % self.frame_rate == 0:
            self.seconds_recorded += 1

        if not hasattr(self, 'recording_start_time'):
            self.recording_start_time = datetime.now()
            self._initialize_video_writer(frame)

        if self.seconds_recorded >= self.video_duration_seconds:
            self.video_writer.release()
            self.recording_start_time = datetime.now()
            self.seconds_recorded = 0
            self._initialize_video_writer(frame)

        self.video_writer.write(frame)

    def _initialize_video_writer(self, frame):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frame_height, frame_width, _ = frame.shape
        start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = os.path.join(self.output_dir, f"{start_time}.avi")
        self.video_writer = cv2.VideoWriter(output_file, fourcc, self.frame_rate, (frame_width, frame_height))

    def stop_recording(self):
        if hasattr(self, 'video_writer'):
            self.video_writer.release()