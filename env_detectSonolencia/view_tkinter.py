import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import dlib
import os
from imutils import face_utils
from scipy.spatial import distance as dist


class WebcamApp:
    def __init__(self, root):
        self._detector = dlib.get_frontal_face_detector()
        self._preditor = dlib.shape_predictor(os.path.join("data", "shape_predictor_68_face_landmarks.dat"))
        self.qtd_frame = 20
        self.contador = 0
        self.root = root
        self.root.title("Webcam App")
        self.root.geometry("800x600")
        self.root.resizable(False, False)

        self.main_frame = tk.Frame(self.root, width=800, height=600)
        self.main_frame.pack()

        self.webcam_frame = tk.Frame(self.main_frame, width=700, height=600)
        self.webcam_frame.pack(side=tk.LEFT)

        self.button_frame = tk.Frame(self.main_frame, width=50, height=100)
        self.button_frame.pack(side=tk.LEFT)

        self.start_button = tk.Button(self.button_frame, text="Start", command=self.start_webcam)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(self.button_frame, text="Stop", command=self.stop_webcam, state=tk.DISABLED)
        self.stop_button.pack()

        self.ear_frame = tk.Frame(self.root, width=400, height=25)
        self.ear_frame.pack()

        self.ear_label = tk.Label(self.ear_frame, text="EAR Value: ", font=("Arial", 20) )
        self.ear_label.pack(pady=10)

        self.video_capture = None
        self.video_display = None
        self.is_running = False

    def start_webcam(self):
        self.video_capture = cv2.VideoCapture(0)
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.show_webcam()

    def stop_webcam(self):
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.video_capture.release()
        self.webcam_frame.config(bg="")

    def calcular_ear(self, olho):
        A = dist.euclidean(olho[1], olho[5])
        B = dist.euclidean(olho[2], olho[4])

        C = dist.euclidean(olho[0], olho[3])
        ear = (A + B) / (2.0 * C)

        return ear

    def show_webcam(self):
        (inicio_esq, fim_esq) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (inicio_dir, fim_dir) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        _, frame = self.video_capture.read()

        if frame is None:
            frame = np.zeros((600, 500, 3), dtype=np.uint8)
            cv2.putText(frame, "No Video", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            self.ear_label.config(text="EAR Value: 0.00")

        frame = cv2.resize(frame, (600, 500))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self._detector(frame, 0)

        for rect in rects:
            shape = self._preditor(frame, rect)
            shape = face_utils.shape_to_np(shape)

            olho_esq = shape[inicio_esq:fim_esq]
            olho_dir = shape[inicio_dir:fim_dir]
            ear_esq = self.calcular_ear(olho_esq)
            ear_dir = self.calcular_ear(olho_dir)

            ear = (ear_esq + ear_dir) / 2.0

            if ear < 0.25:
                self.contador += 1
                if self.contador >=  self.qtd_frame:
                    self.ear_label.config(text="[ALERTA] Possivel estado de sonolencia!", foreground="red")
            else:
                self.contador = 0
                self.ear_label.config(text="EAR Value: {:.2f}".format(ear), foreground="black")


            casco_olho_esq = cv2.convexHull(olho_esq)
            casco_olho_dir = cv2.convexHull(olho_dir)

            cv2.drawContours(frame, [casco_olho_esq], -1, (255, 255, 255), 1)
            cv2.drawContours(frame, [casco_olho_dir], -1, (250, 255, 255), 1)           

        image = Image.fromarray(frame)
        image_tk = ImageTk.PhotoImage(image)

        if self.video_display is None:
            self.video_display = tk.Label(self.webcam_frame, image=image_tk)
            self.video_display.image = image_tk
            self.video_display.pack()
        else:
            self.video_display.configure(image=image_tk)
            self.video_display.image = image_tk

        if self.is_running:
            self.video_display.after(10, self.show_webcam)


if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamApp(root)
    root.mainloop()
