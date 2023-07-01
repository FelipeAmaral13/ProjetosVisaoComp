import cv2
import cv2.aruco as aruco
import numpy as np

# Configuração dos parâmetros dos marcadores ArUcos
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

# Carregar câmera
cap = cv2.VideoCapture(0)

# Matrizes de câmera e coeficientes de distorção (substitua com seus próprios valores)
cameraMatrix = np.array([[480.23597887, 0.0, 324.31561662],
                         [0.0, 482.3132088, 282.66751015],
                         [0.0, 0.0, 1.0]])
distCoeffs = np.array([[0.04415305, -0.53783755, -0.00830731,  0.00804933, -1.24262937]])

while True:
    # Capturar frame da webcam
    ret, frame = cap.read()

    # Converter o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar marcadores ArUcos
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if np.all(ids is not None):
        # Desenhar contornos e IDs dos marcadores na imagem
        aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimar pose dos marcadores
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, distCoeffs)

        # Desenhar eixos 3D para cada marcador
        for i in range(len(ids)):
            aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1)

    # Mostrar imagem resultante
    cv2.imshow('ArUco Pose Estimation', frame)

    # Parar o loop ao pressionar a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

