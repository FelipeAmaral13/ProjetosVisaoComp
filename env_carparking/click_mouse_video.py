# -*- coding: utf-8 -*-
import cv2
import json
import os

point = (0, 0)
point_list = []


class VideoProcessor:
    """
    A class for processing video from a webcam.

    Args:
        video_path (str): Path to the video file.

    Attributes:
        video_path (str): Path to the video file.
        data_json_path (str): Path to the JSON data file.
        click_counter (int): Counter for mouse click events.
        cap (cv2.VideoCapture): VideoCapture object for capturing video frames.

    Methods:
        __init__(self, video_path):
            Initializes a VideoProcessor object.

        mouse_callback(self, event, x, y, flags, param):
            Callback function for mouse events.

        run(self):
            Runs the video processing loop.
    """

    def __init__(self, video_path):
        """
        Initializes a VideoProcessor object.

        Args:
            video_path (str): Path to the video file.
        """
        self.video_path = video_path
        self.data_json_path = os.path.join("json", "data.json")
        self.click_counter = 0
        self.cap = cv2.VideoCapture(self.video_path)
        cv2.namedWindow("Webcam")
        cv2.setMouseCallback("Webcam", self.mouse_callback)
        with open(self.data_json_path, "w") as file:
            file.write("")

    def mouse_callback(self, event, x, y, flags, param):
        global point
        """
        Callback function for mouse events.

        Args:
            event (int): Type of mouse event.
            x (int): X-coordinate of the mouse event.
            y (int): Y-coordinate of the mouse event.
            flags (int): Additional flags for the mouse event.
            param: Additional parameters for the mouse event.
        """
        if event == cv2.EVENT_LBUTTONUP:
            self.click_counter += 1

            point_dict = {"coord": [x, y]}
            point = (x, y)
            point_list.append(point)

            # Serializar o dicionário em uma string JSON
            json_str = json.dumps(point_dict, indent=2)

            with open(self.data_json_path, "a") as file:
                if self.click_counter == 1:
                    # Adicionar colchete abrindo antes do primeiro objeto JSON
                    file.write("[")
                else:
                    # Adicionar uma vírgula antes do objeto JSON, exceto para o primeiro objeto
                    file.write(", ")
                file.write(json_str)

            # if self.click_counter == 4:
            #     with open(self.data_json_path, "a") as file:
            #         # Adicionar colchete fechando após o último objeto JSON
            #         file.write("]")

    def run(self):
        """
        Runs the video processing loop.
        """
        while True:
            ret, frame = self.cap.read()
            # frame = cv2.resize(frame, (960, 540))
            for i in point_list:
                x = i[0]
                y = i[1]
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            if not ret:
                break

            cv2.imshow("Webcam", frame)

            if cv2.waitKey(300) == 27:
                with open(self.data_json_path, "a") as file:
                    # Adicionar colchete fechando após o último objeto JSON
                    file.write("]")
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_processor = VideoProcessor(os.path.join("video", "1.mp4"))
    video_processor.run()
