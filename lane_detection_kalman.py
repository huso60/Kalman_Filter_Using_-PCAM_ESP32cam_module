import numpy as np
import cv2


kalman = cv2.KalmanFilter(4, 2)


kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)


kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)


kalman.measurementNoiseCov = np.array([[1, 0],
                                       [0, 1]], np.float32) * 0.1


kalman.statePre = np.array([[0], [0], [0], [0]], np.float32)
kalman.errorCovPost = np.eye(4, dtype=np.float32)



while True:
    url=r'http://192.168.43.252/capture'
    cap = cv2.VideoCapture(url)
    ret, frame = cap.read()
    if not ret:
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)


    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:

        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])


            measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
            kalman.correct(measurement)


            predicted = kalman.predict()


            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
            cv2.circle(frame, (int(predicted[0]), int(predicted[1])), 10, (0, 0, 255),-1)

    cv2.imshow('Frame', frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
