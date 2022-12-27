# -*- coding: utf-8 -*-
import logging
from queue import Queue
from PySide2 import QtCore
from controller.main_detect_emotion import FaceEmotion
import cv2
import imutils
import numpy as np
from keras.models import model_from_json
from keras.optimizers import SGD
import os

class PipelineManager(QtCore.QThread):
    # sendFrameToView = QtCore.Signal(object)

    def __init__(self, frame_queue: Queue, result_queue: Queue):
        super().__init__()
        self.log = logging.getLogger(__name__)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.model = model_from_json(open(os.path.join(os.getcwd(),'models', 'Face_model_architecture_80_train.json')).read())
        self.model.load_weights(os.path.join(os.getcwd(), 'models', 'Face_model_weights_80_train.h5'))
        self.sgd = SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.sgd)
        self.emotion = FaceEmotion()


    def run(self):

        while not self.isInterruptionRequested():
            frame = self.frame_queue.get()
            video_capture = cv2.VideoCapture(0)

            while True:
            # Captura frame-by-frame
                _, frame = video_capture.read()

                # frame = imutils.resize(frame, width = 800)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detecta as Faces
                gray, detected_faces = self.emotion.detect_face(frame)
                face_index = 0

                # Previsões
                for face in detected_faces:
                    (x, y, w, h) = face
                    if w > 100:
                        # Desenha um retângulo em torno das faces
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Extrai as features
                        extracted_face = self.emotion.extract_face_features(gray, face, (0.075, 0.05)) #(0.075, 0.05)

                        # Prevendo sorrisos
                        prediction_result = self.model.predict(extracted_face.reshape(1, 48, 48, 1))
                        predictions = np.argmax(prediction_result)

                        # Desenha o rosto extraído no canto superior direito
                        frame[face_index * 48: (face_index + 1) * 48, -49:-1, :] = cv2.cvtColor(extracted_face * 255, cv2.COLOR_GRAY2RGB)

                        # Anota a imagem principal com uma etiqueta
                        if predictions == 3:
                            cv2.putText(frame, "Feliz!!",(x,y), cv2.FONT_ITALIC, 2, 155, 10)
                        elif predictions == 0:
                            cv2.putText(frame, "Nervoso",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
                        elif predictions == 1:
                            cv2.putText(frame, "Esnobe",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
                        elif predictions == 2:
                            cv2.putText(frame, "Com Medo",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
                        elif predictions == 4:
                            cv2.putText(frame, "Triste",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
                        elif predictions == 5:
                            cv2.putText(frame, "Surpreso",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
                        else :
                            cv2.putText(frame, "Neutro",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)

                        # Incrementa o contador
                        face_index += 1

                self.result_queue.put(frame)

    def __del__(self):
        try:
            self.requestInterruption()
            self.wait()
        except Exception as erro:
            self.log.debug(f"fails to try to QThread {erro}")
