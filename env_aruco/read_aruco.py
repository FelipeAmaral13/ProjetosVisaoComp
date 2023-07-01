import cv2
import cv2.aruco as aruco

def main():
    # Carregar o arquivo YAML de calibração da câmera
    fs = cv2.FileStorage("calibration.yml", cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()

    # Configuração do dicionário ArUco
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

    # Configuração dos parâmetros do marcador ArUco
    parameters = aruco.DetectorParameters_create()

    # Carrega a imagem da câmera (0 indica a câmera padrão)
    cap = cv2.VideoCapture(0)

    while True:
        # Leitura do quadro da câmera
        ret, frame = cap.read()

        # Correção de distorção da imagem usando os parâmetros da calibração
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

        # Conversão para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecção dos marcadores ArUco
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # Desenhar contornos e IDs dos marcadores detectados
        frame = aruco.drawDetectedMarkers(frame, corners, ids)

        # Exibir a imagem resultante
        cv2.imshow('ArUco Marker Detection', frame)

        # Verificar se a tecla 'q' foi pressionada para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberação dos recursos
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
