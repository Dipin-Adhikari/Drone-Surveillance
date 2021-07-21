import cv2
import numpy as np

cap = cv2.VideoCapture("drone.mp4")
blank = np.zeros((150, 1920))
kernel = np.ones((3, 3), np.uint8)
count = 0

def img_ready(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (7, 7), 200)
    canny_img = cv2.Canny(blur_img, 80, 80)
    dialate_img = cv2.dilate(canny_img, kernel, iterations=1)
    return dialate_img


def get_contours(imgs):
    global count
    contours, hierarchy = cv2.findContours(imgs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5400 and area < 30000:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            count += 1
            cv2.putText(img, str(int(count / 100)), (1320, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
            cv2.putText(img, str(int(area / 100)), (x + 25, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0))
            cv2.putText(img, "Dipin Adhikari", (650, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)


while True:
    success, img = cap.read()
    img = cv2.resize(img, (1500, 900))
    final_img = img_ready(img)
    get_contours(final_img)

    cv2.imshow("Drone Surveillance", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
