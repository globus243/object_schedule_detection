import threading

import cv2

# based on https://stackoverflow.com/a/69141497/898082
class Camera:
    last_frame = None
    last_ready = None
    lock = threading.Lock()

    def __init__(self, rtsp_link):
        self.capture = cv2.VideoCapture(rtsp_link)
        thread = threading.Thread(target=self.rtsp_cam_buffer, name="rtsp_read_thread")
        thread.daemon = True
        thread.start()

    def rtsp_cam_buffer(self):
        while True:
            ret, frame = self.capture.read()
            if not ret:
                print("Frame read failed or stream ended, retrying...")
                continue
            with self.lock:
                self.last_ready, self.last_frame = ret, frame

    def read(self):
        with self.lock:
            if self.last_ready and self.last_frame is not None:
                return True, self.last_frame.copy()
            else:
                return None

    def is_opened(self):
        with self.lock:
            return self.last_ready and self.last_frame is not None
