import numpy as np
import cv2

# Defina as dimensões do tabuleiro de xadrez (número de pontos internos)
num_cols = 9
num_rows = 6

# Crie uma lista para armazenar os pontos do tabuleiro de xadrez
obj_points = []
img_points = []

# Gere as coordenadas dos pontos do objeto (3D) do tabuleiro de xadrez
objp = np.zeros((num_cols * num_rows, 3), np.float32)
objp[:, :2] = np.mgrid[0:num_cols, 0:num_rows].T.reshape(-1, 2)

# Inicialize as variáveis de captura de vídeo da câmera
cap = cv2.VideoCapture(0)
success, frame = cap.read()

while True:
    # Exiba o frame capturado
    cv2.imshow("Captured Frame", frame)
    key = cv2.waitKey(1)

    # Pressione a tecla 's' para salvar os pontos do tabuleiro de xadrez
    if key == ord('s'):
        # Converta o frame para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Encontre os cantos do tabuleiro de xadrez
        ret, corners = cv2.findChessboardCorners(gray, (num_cols, num_rows), None)

        # Se os cantos forem encontrados, adicione os pontos do objeto e da imagem às listas
        if ret == True:
            obj_points.append(objp)
            img_points.append(corners)

            # Desenhe e exiba os cantos do tabuleiro de xadrez
            cv2.drawChessboardCorners(frame, (num_cols, num_rows), corners, ret)
            cv2.imshow("Detected Corners", frame)
            cv2.waitKey(500)

    # Pressione a tecla 'q' para sair do loop de captura
    if key == ord('q'):
        break

    # Capture um novo frame
    success, frame = cap.read()

# Realize a calibração da câmera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

print("Matriz de Câmera:")
print(camera_matrix)
print("\nCoeficientes de Distorção:")
print(dist_coeffs)

# Salve os parâmetros de calibração em um arquivo YAML
calibration_file = "camera_calibration.yml"
calibration_data = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_WRITE)
calibration_data.write("camera_matrix", camera_matrix)
calibration_data.write("dist_coeffs", dist_coeffs)
calibration_data.release()

# Libere os recursos
cap.release()
cv2.destroyAllWindows()
