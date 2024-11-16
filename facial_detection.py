import cv2
import cv2.data

def capture_video():
    # Pegando a câmera, indice represeta qual câmera, que no meu caso é uma só
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Erro ao acessar a webcam")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # face_cascade_front = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            # faces_front = face_cascade_front.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            # faces_profile = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            # faces = list(faces_front) + list(faces_profile)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_video()