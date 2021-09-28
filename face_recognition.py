# importing the libraries
import cv2

# loading classifier
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


# function to detect single image in a frame
def detect(gray, frame):
    # gray to detect image while frame is the live image to draw it

    # get the coordinates of the rectangle to detect the face
    # x, y , w, h
    # x and y are coordinates of upper left corner
    # w = width, h = height

    # to get x,y,w,h
    # 1.3 and 5 is good combo to detect face in webcam
    # 1.3 is the scale, 5 is the number of neighbours in a zone
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # faces contain these 4 variables
    for (x, y, w, h) in faces:
        # x+w y+h to get lower right of the rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # take the gray image which corresponds to the detected face
        # need 2 roi to detect faces both in gray image and color image
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        smile = smile_cascade.detectMultiScale(roi_gray, 1.3, 20)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        for (bx, by, bw, bh) in smile:
            cv2.rectangle(roi_color, (bx, by), (bx+bw, by+bh), (0,0,255), 2)
    return frame


# open webcam
# 0 for internal webcam, 1 for external webcam
video_capture = cv2.VideoCapture(0)

while True:
    # read method returns 2 element, therefore need the '_'
    _, frame = video_capture.read()
    # grayscale image to read
    # COLOR_BGR2GRAY does an average of blue green red to get contrast of black and white
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_detection = detect(gray, frame)
    cv2.imshow('Video', face_detection)
    if cv2.waitKey(1) & 0xFF == ord('f'):
        break

video_capture.release()
cv2.destroyAllWindows()
