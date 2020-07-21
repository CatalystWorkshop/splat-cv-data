import cv2
import sys
from scripts.utils import print

def show_webcam(src):
    cam = cv2.VideoCapture(src)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920);
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080);
    while True:
        ret_val, img = cam.read()
        cv2.imshow('View Webcam/Virtual Camera', img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    print("View Webcam/Virtual Camera")
    print("------")
    virtual_cam_id = 1

    if len(sys.argv) > 1:
        virtual_cam_id = int(sys.argv[1])

    print(f"Using Virtual Camera {virtual_cam_id}")

    show_webcam(virtual_cam_id)

if __name__ == '__main__':
    main()