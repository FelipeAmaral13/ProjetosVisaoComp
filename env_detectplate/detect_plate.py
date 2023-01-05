import cv2
import imutils
import numpy as np
import pickle
from imutils import paths
from tools.license_plate import LicensePlateDetector
from tools.descriptors import NormalizePixels
import os



class DetectPlate():

    def __init__(self):
        self.charModel = pickle.loads(open(os.path.join(os.getcwd(), "models", "simple_char.cpickle"), "rb").read())
        self.digitModel = pickle.loads(open(os.path.join(os.getcwd(), "models", "simple_digit.cpickle"), "rb").read())
        self.size_blocks = ((5, 5), (5, 10), (10, 5), (10, 10))
        self.desc = NormalizePixels(size_blocks=self.size_blocks)



    def read_images(self):
        for imagePath in sorted(list(paths.list_images(os.path.join(os.getcwd(), 'testing_dataset')))):
        
            image = cv2.imread(imagePath)

            if image.shape[1] > 640:
                image = imutils.resize(image, width=640)
            self.locate_plate(image)


    def locate_plate(self, image):

        # Inicializa o detector da placa e detecta as placas e os caracteres
        loc_plate = LicensePlateDetector(image, numChars=7)
        plates = loc_plate.detect_plates()

        for (self.locPlatesBox, self.chars) in plates:
            self.locPlatesBox = np.array(self.locPlatesBox).reshape((-1, 1, 2)).astype(np.int32)
            
            self.text = ""  # Texto dos caracteres reconhecidos

            for (idx, char) in enumerate(self.chars):
                pre_char = LicensePlateDetector.preprocessChar(char)  # Pre-processar os chars
                if pre_char is None:
                    continue
                features = self.desc.describe(pre_char).reshape(1, -1)

                # Classificador de caracteres
                if idx < 3:
                    prediction_char = self.charModel.predict(features)[0]
                
                # Classificador de numeros
                else:
                    prediction_char = self.digitModel.predict(features)[0]

                self.text += prediction_char.upper()

            
            
            self.bbox_chars(self.chars, image)
            
    
    def bbox_chars(self, chars, image):
        if len(chars) > 0:
            # Calcula o centro da caixa delimitadora da placa
            M = cv2.moments(self.locPlatesBox)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Desenha a região da placa e o texto da placa na imagem
            cv2.drawContours(image, [self.locPlatesBox], -1, (0, 255, 0), 2)
            cv2.putText(image, self.text, (cX - (cX // 5), cY - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 0, 255), 2)

            cv2.imshow("Imagem Original", image)
            cv2.waitKey(0)
 
if __name__ == "__main__":

    d = DetectPlate()
    d.read_images()