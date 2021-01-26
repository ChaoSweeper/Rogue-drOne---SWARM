# Import required libraries
import cv2

# Load the cascade model
cars_model = cv2.CascadeClassifier("car.xml")

""" 
Below is the syntax for using detectMultiScale() method in openCV
to detect the coordinates of vehicle's in a video.
cars = cars_model.detectMultiScale(frame, scaleFactor, minNeighbors)
cv2.rectangle(frame, point1, point2, color(), thickness=value) 
"""


def detect_vehicle(frame):
    cars = cars_model.detectMultiScale(frame, 1.15, 4)
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 255), thickness=3)


def detecter():
    vehicle_video = cv2.VideoCapture("cars_test1.mp4")

    while vehicle_video.isOpened():
        ret, frame = vehicle_video.read()
        control_key = cv2.waitKey(1)

        if ret:
            vehicles_frame = detect_vehicle(frame)
            cv2.imshow("frame", vehicles_frame)
        else:
            break
        if control_key == ord("w"):
            break

    vehicle_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detecter()
