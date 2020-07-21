import cv2
import threading
import time

class VideoStreamWidget(object):
    def __init__(self, src, resolution='720p'):
        self.capture = cv2.VideoCapture(src)
        if resolution == '1080p':
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920);
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080);
        else:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280);
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720);
        # Start the thread to read frames from the video stream
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.frame = None
        self.status = True
        self.thread.start()
        self.now = time.time()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                self.now = time.time()
            time.sleep(.01)
    def finish(self):
        self.capture.release()
        self.thread.join()