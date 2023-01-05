# Projeto Detecção de Emoções

Objetivo desse projeto é através de uma interface gráfica, acessar a webcam e fazer a detecção de emoções do vídeo.

![gui](https://user-images.githubusercontent.com/5797933/209832167-a1a49246-7dd1-425d-8904-7fd941b2f312.PNG)

A interface gráfica possui dois botões:

- Start Video: Ligar a webcam
- Stop Video: Desligar a webcam

Depois que o streaming de video é ligado, primeiramente detecta-se a face humana por uso de Haarcascades, logo após o classificador treinado é responsável por fazer classificação da emoção detectada na face.

