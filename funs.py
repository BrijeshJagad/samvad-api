import cv2
import numpy as np
import mediapipe as mp

# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import LSTM, Dense, GRU, Dropout
import onnxruntime as ort


def create_model(model_path):
    providers = ['CPUExecutionProvider']
    model = ort.InferenceSession(model_path, providers=providers)
    
    return model

class Predictor:
    def __init__(self, model_path):
        self.model = create_model(model_path)
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils 
        self.actions = ['arm', 'drink', 'computer', 'before', 'go', 'who', 'a', 'b', 'c', 'd', 'e','f','g','h','i','j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's']
        self.threshold = 0.8
        self.counts = []

    def draw_styled_landmarks(self,image, results):
        # Draw face connections
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
                                  self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )
        # Draw pose connections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                  self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                  )
        # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                  self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                  )
        # Draw right hand connections
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                  self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )
    
    def mediapipe_detection(self, image, holistic):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = holistic.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results
    
    def draw_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION) # Draw face connections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS) # Draw pose connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
    
    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

    def most_frequent(self, List):
        return max(set(List), key = List.count)

    def predict(self, video_path):
        sequence = []
        sentence = []
        cap = cv2.VideoCapture(video_path)

        with self.mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
            while(True):

                ret, frame = cap.read()
                if ret == False:
                    break

                # Make detections
                image, results = self.mediapipe_detection(frame, holistic)
                # print(results)

                # Draw landmarks
                self.draw_styled_landmarks(image, results)

                # 2. Prediction logic
                keypoints = self.extract_keypoints(results)
        #         sequence.insert(0,keypoints)
        #         sequence = sequence[:30]
                sequence.append(keypoints)
                sequence = sequence[-20:]

                if len(sequence) == 20:
                    # onnx_pred = m.run('dense_20', {"input": sequence}
                    # x = np.expand_dims(sequence, axis=0)
                    # print(x.shape)
                    res = self.model.run(['dense_20'],{'input': np.expand_dims(sequence, axis=0).astype('float32')})[0][0]


                # 3. Viz logic
                    if res[np.argmax(res)] > self.threshold:
                        self.counts.append(self.actions[np.argmax(res)])
                        if len(sentence) > 0:
                            if self.actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(self.actions[np.argmax(res)])
                        else:
                            sentence.append(self.actions[np.argmax(res)])

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()

        if len(self.counts):
            return self.most_frequent(self.counts)

        return "Couldn't recognize!"
