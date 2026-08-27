from __future__ import annotations

import cv2


class Camera:
    def __init__(self, index=0, width=1280, height=720, fps=30):
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self.capture = None

    def open(self):
        self.capture = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = cv2.VideoCapture(self.index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.index}")
        return self

    def read(self):
        if self.capture is None:
            raise RuntimeError("Camera is not open")
        return self.capture.read()

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.release()

