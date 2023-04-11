import cv2
import os
from datetime import datetime


class VideoRecorder:
    def __init__(self, video_duration, output_dir='Recordings'):
        if video_duration < (5 / 60) or video_duration > 59:
            raise ValueError("Video duration must be between 5 seconds and 59 minutes.")

        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.video_duration_seconds = video_duration * 60
        # self.frame_rate = 29
        self.frames_received = 0
        self.seconds_recorded = 0

    def record(self, frame, fps):
        self.frames_received += 1

        if self.frames_received % fps == 0:
            self.seconds_recorded += 1

        if not hasattr(self, 'recording_start_time'):
            self.recording_start_time = datetime.now()
            self._initialize_video_writer(frame, fps)

        if self.seconds_recorded >= self.video_duration_seconds:
            self.video_writer.release()
            self.recording_start_time = datetime.now()
            self.seconds_recorded = 0
            self._initialize_video_writer(frame, fps)

        self.video_writer.write(frame)

    def _initialize_video_writer(self, frame, fps):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frame_height, frame_width, _ = frame.shape
        start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = os.path.join(self.output_dir, f"{start_time}.avi")
        self.video_writer = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    def stop_recording(self):
        if hasattr(self, 'video_writer'):
            self.video_writer.release()


if __name__ == "__main__":
    input_source = 0  # Change this to your input source, e.g. camera, video file, IP cam, RTP stream, etc.
    video_duration = 10  # Video duration in minutes (min: 5 seconds, max: 59 minutes)

    recorder = VideoRecorder(video_duration)
    capture = cv2.VideoCapture(input_source)
    frame_rate = capture.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        recorder.record(frame, frame_rate)
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    recorder.stop_recording()
    capture.release()
    cv2.destroyAllWindows()