# -*- coding: utf-8 -*-
import logging
from PySide2 import QtGui, QtWidgets
from PySide2.QtCore import Slot, Qt

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.win_table = None
        self.log = logging.getLogger(__name__)
        title = "Face Emotion"
        self.setObjectName("MainWindow")
        self.setWindowTitle(title)
        self.resize(450, 450)

        main_layout = QtWidgets.QVBoxLayout()
        self.header_layout = QtWidgets.QHBoxLayout()
        self.body_layout = QtWidgets.QGridLayout()

        header_widget = QtWidgets.QWidget()
        header_widget.setLayout(self.header_layout)

        body_widget = QtWidgets.QWidget()
        body_widget.setLayout(self.body_layout)

        self.button_start_video = QtWidgets.QPushButton("Start video ")
        self.button_start_video.setObjectName("button_video_start")

        self.button_stop_video = QtWidgets.QPushButton("stop video")
        self.button_stop_video.setObjectName("button_video_stop")

        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.addWidget(self.button_start_video)
        self.button_layout.addWidget(self.button_stop_video)

        main_layout.addWidget(header_widget)
        main_layout.addWidget(body_widget)
        main_layout.addLayout(self.button_layout)

        title_widget = QtWidgets.QLabel(title)
        title_widget.setStyleSheet('font-family: "DejaVu Sans"; font-size: 12pt')

        self.graphics_view = QtWidgets.QGraphicsView()
        self.scene = QtWidgets.QGraphicsScene()
        self.graphics_view.setScene(self.scene)

        self.body_layout.addWidget(self.graphics_view)

        widget = QtWidgets.QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)


    @Slot()
    def show_frame(self, frame):
        self.scene.clear()
        q_image = QtGui.QImage(
            frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888
        )
        pixmap = QtGui.QPixmap.fromImage(q_image.rgbSwapped())
        pixmap_item = self.scene.addPixmap(pixmap)
        self.graphics_view.width()
        self.graphics_view.height()
        self.graphics_view.fitInView(pixmap_item, Qt.KeepAspectRatio)


# if __name__ == "__main__":
#     import sys

#     app = QtWidgets.QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     app.exec_()