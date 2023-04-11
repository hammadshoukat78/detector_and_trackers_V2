from Utils.ObjectTrackers.ByteTrack.tracker import ByteTracker
from Utils.ObjectTrackers.DeepSort.tracker import DeepSortTracker


# byte_track_obj = ByteTracker()
deep_sort_obj = DeepSortTracker()

deep_sort_obj.process_video(source="trafic.mp4")