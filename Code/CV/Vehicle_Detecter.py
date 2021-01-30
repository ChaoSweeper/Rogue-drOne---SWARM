# Import required libraries
import cv2 as cv
import export

# Load the cascade model
cars_model = cv.CascadeClassifier("car.xml")

""" 
Below is the syntax for using detectMultiScale() method in openCV
to detect the coordinates of vehicle's in a video.
cars = cars_model.detectMultiScale(frame, scaleFactor, minNeighbors)
cv2.rectangle(frame, point1, point2, color(), thickness=value) 
"""


def detect_vehicle(frame):
    cars = cars_model.detectMultiScale(frame, 1.15, 4)
    for (x, y, w, h) in cars:
        cv.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 255), thickness=2)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, "Vehicle", (x + 6, y - 6), font, 0.5, (0, 255, 0), 1)
    return frame


def detecter():
    # Import the video for detection
    vehicle_video = cv.VideoCapture("cars_test1.mp4")

    # Create frame width & height variables
    frame_width = int(vehicle_video.get(3))
    frame_height = int(vehicle_video.get(4))

    # Create the format to save the video in
    video_format = cv.VideoWriter_fourcc(*"mp4v")

    # Create a frame rate variable
    frame_rate = 10.0

    out = cv.VideoWriter(
        "Car Detection Results.mp4",
        video_format,
        frame_rate,
        (frame_width, frame_height),
        True,
    )

    while vehicle_video.isOpened():
        ret, frame = vehicle_video.read()

        if ret == True:
            vehicles_frame = detect_vehicle(frame)
            cv.imshow("frame", vehicles_frame)
            out.write(vehicles_frame)

            if cv.waitKey(1) & 0xFF == ord("w"):
                break
        else:
            break
    export()
    vehicle_video.release()
    out.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    detecter()
