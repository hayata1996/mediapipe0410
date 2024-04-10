import cv2
import mediapipe as mp

# MediaPipeのセットアップ
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# ウェブカメラの準備
cap = cv2.VideoCapture(0)

# FaceMeshの設定
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # BGR画像をRGBに変換
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 検出を実行
        results = face_mesh.process(image)

        # RGB画像をBGRに戻す
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        # 結果を描画
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION)

        # 画像を表示
        cv2.imshow('MediaPipe FaceMesh', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
