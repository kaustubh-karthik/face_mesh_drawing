import cv2
import mediapipe as mp
import numpy as np
import time
class detector():
    def __init__(self, detection, mode = False, detection_conf = 0.5, track_conf = 0.5):

        detector_object = {
            'hands': [mp.solutions.hands, mp.solutions.hands.Hands],
            'pose': [mp.solutions.pose, mp.solutions.pose.Pose],
            'face_mesh': [mp.solutions.face_mesh, mp.solutions.face_mesh.FaceMesh]
            }

        self.object_draw_details = {
            'hands': ['results.multi_hand_landmarks', 'self.mp_object.HAND_CONNECTIONS'],
            'pose': ['results.pose_landmarks', 'self.mp_object.POSE_CONNECTIONS'],
            'face_mesh': ['results.multi_face_landmarks', 'self.mp_object.FACEMESH_TESSELATION']}

        self.object_iteration = {
            'hands': 'results.multi_hand_landmarks',
            'pose': '[results.pose_landmarks]',
            'face_mesh': 'results.multi_face_landmarks'
        }

        self.detection = detection

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_object = detector_object[self.detection][0]
        self.object = detector_object[self.detection][1]()

        self.results_check = self.object_draw_details[detection][0]
        self.connection_details = self.object_draw_details[detection][1]
        self.iteration = self.object_iteration[detection]

        self.prev_time = 0


    def find_lms(self, img, draw = True, lm_id = None):
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.object.process(rgb_img)

        lm_list = []

        if eval(self.results_check):

            for lms in eval(self.iteration):

                self.mp_draw.draw_landmarks(img, lms, eval(self.connection_details))

                for id, lm in enumerate(lms.landmark):
                    
                    height, width, channel = img.shape
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    lm_list.append([id, cx, cy])

                    if draw:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 0))

        if lm_id != None:
            print(lm_list[lm_id])

        return img, lm_list

    def display_img(self, img, fps = True):

        if fps:
            curr_time = time.time()
            fps = 1/(curr_time - self.prev_time)
            self.prev_time = curr_time


        cv2.putText(
            img,
            text = str(int(fps)),
            org = (10, 70),
            fontFace = cv2.FONT_HERSHEY_COMPLEX,
            fontScale = 3,
            color = (150, 150, 150),
            thickness = 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


def main():

    # Choose one out of 'hands, pose, face_mesh'
    detect = detector('face_mesh')

    cap = cv2.VideoCapture(1) # Check for error

    while True:
        success, img = cap.read()

        img, lm_list = detect.find_lms(img, draw=False)

        detect.display_img(img)

if __name__ == '__main__':
    main()
