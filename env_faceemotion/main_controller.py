# -*- coding: utf-8 -*-
import logging
import sys
from PySide2 import QtWidgets
from queue import Queue
from model.camera import Camera
from model.data_handler import DataHandler
from model.pipeline_manager import PipelineManager
from view import MainWindow


class MainController:
    def __init__(self, video):
        self.log = logging.getLogger(__name__)
        self.frame_queue = Queue(maxsize=50)
        self.result_queue = Queue(maxsize=50)
        self.camera = Camera(camera_id=video, frame_queue=self.frame_queue)
        self.face_emotion = PipelineManager(self.frame_queue, self.result_queue)

        self.window = MainWindow()
        self.data_handler = DataHandler(self.result_queue)

        self.window.button_start_video.clicked.connect(self.start)
        self.window.button_stop_video.clicked.connect(self.stop)

    def run(self):
        self.log.debug("Function run -> Starting MainController...")
        self.data_handler.sendFrameToView.connect(self.window.show_frame)
        self.window.show()

    def start(self):
        self.face_emotion = PipelineManager(self.frame_queue, self.result_queue)
        ## Tudo em threads, uso start pra rodar o run
        self.log.debug("Function start ->starting main controller runs")
        self.data_handler.start()
        self.face_emotion.start()
        self.camera.start()

    def stop(self):
        self.face_emotion.__del__()
        self.camera.stop()
        self.log.debug("Function stop -> stoped")


if __name__ == "__main__":

    qt_app = QtWidgets.QApplication([])
    controller = MainController(video=0)
    controller.run()
    sys.exit(qt_app.exec_())