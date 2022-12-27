# -*- coding: utf-8 -*-
import time
import logging
from typing import Any
from queue import Queue
import cv2
from PySide2 import QtCore


class Camera(QtCore.QThread):
    def __init__(self, camera_id, frame_queue: Queue):
        super().__init__()
        self.log = logging.getLogger(__name__)
        self.frame_queue: Queue = frame_queue
        self.camera_id = camera_id
        self.started = False

    @staticmethod
    def connect_cam(camera_id) -> Any:
        capture = cv2.VideoCapture(camera_id)
        if capture.isOpened():
            return capture
        raise Exception("Camera can't connect")

    def run(self) -> None:
        capture = self.connect_cam(self.camera_id)
        self.started = True
        while (
            capture.isOpened() and not self.isInterruptionRequested() and self.started
        ):
            _, frame = capture.read()
            if frame is not None:
                self.frame_queue.put(frame)
            else:
                raise Exception("Frame is None")

            time.sleep(1 / 30)
        capture.release()

    def stop(self):
        self.started = False


    def __del__(self):
        try:
            self.requestInterruption()
            self.wait()
        except Exception as erro:
            self.log.warning(f"fails to try to QThread {erro} ")
