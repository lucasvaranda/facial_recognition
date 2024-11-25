import cv2
import mediapipe as mp
import os
from tqdm import tqdm

def detect_pose(video_path, output_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erro ao executar o vídeo")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    arm_up = False
    arm_movements_count = 0

    def is_arm_up(landmarks):
        left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
        right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

        left_arm_up = left_elbow.y < left_eye.y
        right_arm_up = right_elbow.y < right_eye.y

        return left_arm_up and right_arm_up

        # left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        # left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        # right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        # right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y


    for _ in tqdm(range(total_frames), desc='Processando vídeo'):
        ret, frame = cap.read()

        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if is_arm_up(results.pose_landmarks.landmark):
                if not arm_up:
                    arm_up = True
                    arm_movements_count += 1
                else:
                    arm_up = False
                
                cv2.putText(frame, f'Movimento dos braços: {arm_movements_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'video_pose.mp4')
output_video_path = os.path.join(script_dir, 'output_video_pose_arm_up.mp4')

detect_pose(input_video_path, output_video_path)