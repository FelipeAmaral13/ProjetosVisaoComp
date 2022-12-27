# -*- coding: utf-8 -*-
import logging
from queue import Queue
from PySide2 import QtCore


class DataHandler(QtCore.QThread):
    sendFrameToView = QtCore.Signal(object)

    def __init__(self, frame_queue: Queue) -> None:
        super().__init__()
        self.log = logging.getLogger(__name__)
        self.frame_queue: Queue = frame_queue

    def run(self) -> None:
        self.log.debug("start data handler")
        while not self.isInterruptionRequested():
            frame = self.frame_queue.get()
            self.sendFrameToView.emit(frame)

    def __del__(self):
        try:
            self.sendFrameToView.disconnect()
        except Exception as error:
            self.log.warning(f"Warning: {error}")

        finally:
            try:
                self.requestInterruption()
                self.wait()
            except Exception as error:
                self.log.warning(error)
