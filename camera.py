import numpy as np
import cv2

cap = cv2.VideoCapture(1)


def get_image():
    # Capture frame-by-frame
    _, frame = cap.read()

    # Our operations on the frame come here
    return frame


def get_gary_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), 0)
    return gray


baseline_image = get_gary_img(get_image())

while(True):
    img = get_image()
    gray_frame = get_gary_img(img)

    # Calculating the difference and image thresholding
    delta = cv2.absdiff(baseline_image, gray_frame)
    threshold = cv2.threshold(delta, 35, 255, cv2.THRESH_BINARY)[1]
    # Finding all the contours
    (contours, _) = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Drawing rectangles bounding the contours (whose area is > 5000)
    for contour in contours:
        if cv2.contourArea(contour) < 5000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Display the resulting frame
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()