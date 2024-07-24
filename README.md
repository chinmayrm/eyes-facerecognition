# Face and Eye Detection using OpenCV

This project demonstrates how to perform face and eye detection using OpenCV with Haar cascades.

## Requirements

- Python 3.x
- OpenCV 4.x
- Haar Cascade XML files for face and eye detection (`haarcascade_frontalface_default.xml` and `haarcascade_eye.xml`)

## Installation

1. Install Python 3.x from the [official website](https://www.python.org/).
2. Install OpenCV by running the following command in your terminal:
   ```bash
   pip install opencv-python
   ```
3. Download the Haar Cascade XML files for face and eye detection:
   - [haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
   - [haarcascade_eye.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye.xml)

## Usage

1. Place the Haar Cascade XML files (`haarcascade_frontalface_default.xml` and `haarcascade_eye.xml`) in the same directory as your script.
2. Save the following code in a file named `face_eye_detection.py`:

    ```python
    import cv2
    import numpy as np

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.putText(img, 'face', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in eyes:
                cv2.putText(roi_color, 'eye', (ex, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 255), 2, cv2.LINE_AA)
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

        cv2.imshow('Face & Eye Detection', img)

        if cv2.waitKey(1) & 0xFF == ord('w'):
            break

    cap.release()
    cv2.destroyAllWindows()
    ```

3. Run the script using Python:
   ```bash
   python face_eye_detection.py
   ```
4. The webcam will open and start detecting faces and eyes. Press the 'w' key to exit the program.

## Notes

- Ensure your webcam is working properly before running the script.
- Adjust the scale factors and minimum neighbors in the `detectMultiScale` function if needed to improve detection accuracy.

---

This README provides a clear and concise guide on setting up and running the face and eye detection script using OpenCV.
