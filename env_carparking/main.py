import cv2
import supervision as sv
from ultralytics import YOLO
import os
import torch
import json
import numpy as np

class ParkCarDetect:
    """
    Classe para detectar carros estacionados em uma vaga de estacionamento.
    """
    def __init__(self) -> None:
        """
        Inicializa a classe.

        Carrega o modelo de detecção YOLOv8 e o polígono que representa a vaga de estacionamento.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", device)

        self.green_color = (0, 255, 0)
        self.red_color = (0, 0, 255)  
        self.model = YOLO(os.path.join("model", "yolov8l.pt"))
        self.json_path = os.path.join("json", "data.json")
        self.polygon = self.read_json()
        self.counter = 0

    def read_json(self):
        """
        Lê o polígono que representa a vaga de estacionamento a partir de um arquivo JSON.

        Retorna um array numpy com os pontos do polígono.
        """
        with open(self.json_path, "r", encoding="utf8") as file:
            json_data = json.load(file)

        lista_poligono = []
        temp = []
        for idx, dados in enumerate(json_data):
            temp.append(dados["coord"])
            if (idx + 1) % 4 == 0:
                lista_poligono.append(temp)
                temp=[]
        polygon = np.asarray(lista_poligono)

        return polygon

    def run(self):
        """
        Executa o processamento de vídeo com a detecção de carros e a marcação nos polígonos de interesse.

        O vídeo é lido a partir de um arquivo de vídeo utilizando o OpenCV e processado frame a frame.
        Para cada frame, o modelo de detecção é aplicado e as detecções de carros são obtidas.
        Em seguida, é verificado se as detecções estão dentro dos polígonos de interesse e, se estiverem,
        é desenhada uma marcação na imagem indicando a detecção de carro.
        A imagem processada é exibida em uma janela do OpenCV.

        """
        
        cap = cv2.VideoCapture(os.path.join("video", "1.mp4"))

        while True:
            ret, frame = cap.read()
            

            if not ret:
                break

            results = self.model(
                frame,
                imgsz=[480, 640],
                device="0",
                conf=0.3,
                classes=[67],
                line_thickness=1,
                show=False,
                agnostic_nms=True,
            )[0]

            for polygon in self.polygon:
                cv2.polylines(frame, [polygon], isClosed=True, color=self.green_color, thickness=2)

            self.counter = 0

            detections = sv.Detections.from_yolov8(results)
            for detection in detections:
                x1, y1, x2, y2 = detection[0]
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1
                ponto_central = (int(x + w / 2), int(y + h / 2))
                for polygon in self.polygon:
                    resultado = cv2.pointPolygonTest(polygon, ponto_central, False)
                    if resultado >= 0:
                        self.counter += 1
                        cv2.putText(frame, "Car_Detect", ponto_central, cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.red_color, 1)

            vagas_livres = len(self.polygon) - self.counter

            img_text = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            cv2.putText(
                img_text, f'Total de Vagas Detectadas: {len(self.polygon)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                img_text, f'Total de Carros Detectados em vagas: {self.counter}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                img_text, f'Total de Vagas livres: {vagas_livres}', (50, 250), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2, cv2.LINE_AA)

            img_total = cv2.hconcat([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), cv2.cvtColor(img_text, cv2.COLOR_BGR2RGB)])


            cv2.imshow("Video", img_total)
            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    pcd = ParkCarDetect()
    pcd.run()