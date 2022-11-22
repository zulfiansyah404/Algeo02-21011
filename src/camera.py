import cv2

def capture_image():

    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        cv2.imshow("camera", frame)

        k = cv2.waitKey(1)
        if k == 32:
            img_name = "potret.jpg"
            cv2.imwrite(img_name, frame)
            break
    cam.release()

    cv2.destroyAllWindows()