import cv2 as cv
import os
import shutil


def get_image():
    cam = cv.VideoCapture("Project_DONTUPLOAD\Good Code\Car Detection Results.mp4")
    cur_frame = 0
    make_dict()
    while True:
        ret, frame = cam.read()
        if ret:
            cam.set(cv.CAP_PROP_POS_MSEC, (cur_frame * 500))
            name = "./data/frame" + str(cur_frame) + ".jpg"
            cv.imwrite(name, frame)
            cur_frame += 1
        else:
            break
    cam.release()
    cv.destroyAllWindows()


def make_dict():
    try:
        path = "data"
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)
            os.makedirs(path)
    except OSError:
        print("Error creating directory")


if __name__ == "__main__":
    get_image()