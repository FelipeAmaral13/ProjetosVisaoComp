import cv2 as cv
from cv2 import aruco
import numpy as np
import os


class ArucoDetect:
    def __init__(self, image_dir):
        self.images_list = self.read_images(image_dir)
        self.marker_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.param_markers = aruco.DetectorParameters_create()

    @staticmethod
    def read_images(dir_path):
        img_list = []
        files = os.listdir(dir_path)
        for file in files:
            img_path = os.path.join(dir_path, file)
            image = cv.imread(img_path)
            img_list.append(image)
        return img_list

    def image_augmentation(self, frame, src_image=None, dst_points=None):
        src_h, src_w = frame.shape[:2]
        if src_image is not None:
            src_h, src_w = src_image.shape[:2]

        frame_h, frame_w = frame.shape[:2]
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        src_points = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]])
        H, _ = cv.findHomography(srcPoints=src_points, dstPoints=dst_points)
        warp_image = cv.warpPerspective(src_image, H, (frame_w, frame_h)) if src_image is not None else frame
        cv.fillConvexPoly(mask, dst_points, 255)
        cv.bitwise_and(warp_image, warp_image, frame, mask=mask)

        if src_image is None:
            cv.putText(frame, 'No_Motors_Infos', (dst_points[0][0], dst_points[0][1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    def run(self):
        cap = cv.VideoCapture(0)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                marker_corners, marker_IDs, reject = aruco.detectMarkers(
                    gray_frame, self.marker_dict, parameters=self.param_markers
                )
                if marker_corners:
                    for ids, corners in zip(marker_IDs, marker_corners):
                        corners = corners.reshape(4, 2)
                        corners = corners.astype(int)
                        if ids[0] < len(self.images_list):
                            self.image_augmentation(frame, self.images_list[ids[0]], corners)
                        else:
                            self.image_augmentation(frame, None, corners)
                cv.imshow("frame", frame)
                key = cv.waitKey(1)
                if key == ord("q"):
                    break
        finally:
            cap.release()
            cv.destroyAllWindows()


if __name__ == "__main__":
    image_directory = os.path.join("images", "augmentation")
    aruco_detector = ArucoDetect(image_directory)
    aruco_detector.run()
