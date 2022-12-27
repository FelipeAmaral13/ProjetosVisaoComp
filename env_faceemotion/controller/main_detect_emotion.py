import numpy as np
import cv2
import imutils
from keras.models import model_from_json
from keras.optimizers import SGD
from scipy.ndimage import zoom
import os
import pathlib


class FaceEmotion:

    def __init__(self) -> None:
        # self.model = model_from_json(open(os.path.join(os.getcwd(),'models', 'Face_model_architecture_80_train.json')).read())
        # self.model.load_weights(os.path.join(os.getcwd(), 'models', 'Face_model_weights_80_train.h5'))
        # self.sgd = SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        # self.model.compile(loss='categorical_crossentropy', optimizer=self.sgd)
        self.cascPath = os.path.join(os.getcwd(), 'models', 'haarcascade_frontalface_default.xml')
        self.faceCascade = cv2.CascadeClassifier(self.cascPath)



    def extract_face_features(self, gray, detected_face, offset_coefficients):
        (x, y, w, h) = detected_face
        horizontal_offset = np.int(np.floor(offset_coefficients[0] * w))
        vertical_offset = np.int(np.floor(offset_coefficients[1] * h))

        extracted_face = gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]
        new_extracted_face = zoom(extracted_face, (48. / extracted_face.shape[0], 48. / extracted_face.shape[1]))
        new_extracted_face = new_extracted_face.astype(np.float32)
        new_extracted_face /= float(new_extracted_face.max())

        return new_extracted_face


    def detect_face(self, frame):

        faceCascade = cv2.CascadeClassifier(self.cascPath)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        
        return gray, detected_faces

    # def run (self):

    #     video_capture = cv2.VideoCapture(0)

    #     while True:
    #         # Captura frame-by-frame
    #         ret, frame = video_capture.read()

    #         frame = imutils.resize(frame, width = 800)
    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #         # Detecta as Faces
    #         gray, detected_faces = self.detect_face(frame)

    #         face_index = 0

    #         # Previsões
    #         for face in detected_faces:
    #             (x, y, w, h) = face
    #             if w > 100:
    #                 # Desenha um retângulo em torno das faces
    #                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
    #                 # Extrai as features
    #                 extracted_face = self.extract_face_features(gray, face, (0.075, 0.05)) #(0.075, 0.05)

    #                 # Prevendo sorrisos
    #                 prediction_result = self.model.predict(extracted_face.reshape(1, 48, 48, 1))
    #                 predictions = np.argmax(prediction_result)

    #                 # Desenha o rosto extraído no canto superior direito
    #                 frame[face_index * 48: (face_index + 1) * 48, -49:-1, :] = cv2.cvtColor(extracted_face * 255, cv2.COLOR_GRAY2RGB)

    #                 # Anota a imagem principal com uma etiqueta
    #                 if predictions == 3:
    #                     cv2.putText(frame, "Feliz!!",(x,y), cv2.FONT_ITALIC, 2, 155, 10)
    #                 elif predictions == 0:
    #                     cv2.putText(frame, "Nervoso",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
    #                 elif predictions == 1:
    #                     cv2.putText(frame, "Esnobe",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
    #                 elif predictions == 2:
    #                     cv2.putText(frame, "Com Medo",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
    #                 elif predictions == 4:
    #                     cv2.putText(frame, "Triste",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
    #                 elif predictions == 5:
    #                     cv2.putText(frame, "Surpreso",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
    #                 else :
    #                     cv2.putText(frame, "Neutro",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)


    #                 # Incrementa o contador
    #                 face_index += 1

    #     return frame

