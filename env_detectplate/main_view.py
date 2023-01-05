import sys
import os
from PySide2 import QtWidgets, QtGui, QtCore


class MainWindow(QtWidgets.QWidget):

    loaded = QtCore.Signal(str, QtGui.QImage)

    def __init__(self):
        super().__init__()
        self.title = 'Plate Car Detect'
        self.setWindowTitle(self.title)
        self.resize(450, 450)

        main_layout =  QtWidgets.QVBoxLayout()
        head_layout = QtWidgets.QVBoxLayout()
        body_layout = QtWidgets.QHBoxLayout()

        layout_btn = QtWidgets.QVBoxLayout()
        self.btn = QtWidgets.QPushButton("Upload Image")
        self.btn.clicked.connect(self.get_img)
        layout_btn.addWidget(self.btn)

        layout_vertical_crop = QtWidgets.QVBoxLayout()
        self.label1 =  QtWidgets.QLabel("1 - Plate Region with Perspective Transform")
        self.label2 =  QtWidgets.QLabel("2 - License Plate Threshold - Resized")
        self.label3 =  QtWidgets.QLabel("3 - Original Candidates")
        self.label4 =  QtWidgets.QLabel("4 - Pruned Candidates")
        self.label5 =  QtWidgets.QLabel("5 - Char Threshold")

        layout_vertical_crop.addWidget(self.label1)
        layout_vertical_crop.addWidget(self.label2)
        layout_vertical_crop.addWidget(self.label3)
        layout_vertical_crop.addWidget(self.label4)
        layout_vertical_crop.addWidget(self.label5)


        layout_vertical_original_img = QtWidgets.QVBoxLayout()
        self.label6 =  QtWidgets.QLabel("Original Image Car")
        
        layout_vertical_original_img.addWidget(self.label6)

        head_layout.addLayout(layout_btn)
        body_layout.addLayout(layout_vertical_crop)
        body_layout.addLayout(layout_vertical_original_img)

        main_layout.addLayout(head_layout)
        main_layout.addLayout(body_layout)
        self.setLayout(main_layout)


    def get_img(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, "Open File", os.getcwd(), '*.png, *.jpg;;PNG Files (*.png, *.jpg)')
        self.path_img = fname
        self.pixmap = QtGui.QPixmap(fname[0]).scaled(400, 400,  QtCore.Qt.KeepAspectRatio)
        if self.pixmap.isNull():
            error_message = QtWidgets.QErrorMessage(self)
            error_message.setWindowTitle("Erro Image")
            error_message.showMessage(
                """Please, check if select real image"""
            )
        self.label6.setPixmap(self.pixmap)
        img = QtGui.QImage(fname[0])
        self.loaded.emit(fname[0], img)
    


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())