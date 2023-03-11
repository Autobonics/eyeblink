import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt


def _calc_EAR(lmk):
    # EAR Algorithm
    # required 6 coordinates
    # EAR= ||p2-p6||+||p3-p5|| / 2||p1-p4||
    # || => l2 norm
    [p1, p2, p3, p4, p5, p6] = lmk
    return (np.linalg.norm(p2-p6)+np.linalg.norm(p3-p5))/(2*np.linalg.norm(p1-p4))


def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while cap.isOpened():
            response, image = cap.read()
            if not response:
                # print("End of vid")
                break
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(image)
            if res.multi_face_landmarks:
                landmarks = res.multi_face_landmarks[0].landmark
                # get Eye coordinates
                # src : https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
                # img ref : https://github.com/google/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

                # EAR Algorithm
                # required 6 coordinates
                # EAR= ||p2-p6||+||p3-p5|| / 2||p1-p4||
                # Required coordinates for each eye
                # Left Eye
                #  [lf_p1_ind, lf_p2_ind, lf_p3_ind, lf_p4_ind, lf_p5_ind,
                #   lf_p6_ind]
                lf_ind_map = [362, 385, 387, 263, 373, 380]
                # Right Eye
                rt_ind_map = [33, 160, 158, 133, 153, 144]
                # [rt_p1_ind, rt_p2_ind, rt_p3_ind, rt_p4_ind, rt_p5_ind,
                #               rt_p6_ind]
                lf_lmk = list(map(
                    lambda ind: np.array(
                        [landmarks[ind].x, landmarks[ind].y, landmarks[ind].z]),
                    lf_ind_map
                ))
                rt_lmk = list(map(
                    lambda ind: np.array(
                        [landmarks[ind].x, landmarks[ind].y, landmarks[ind].z]),
                    rt_ind_map
                ))
                lf_EAR = _calc_EAR(lf_lmk)
                rt_EAR = _calc_EAR(rt_lmk)
                # EAR = arverage of lf_EAR and rt_EAR
                # EAR = arverage of lf_EAR and rt_EAR
                EAR = (lf_EAR+rt_EAR)/2
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = cv2.flip(image, 1)
                drowzy_state = "drowzy" if EAR < 0.2 else "active"
                image = cv2.putText(image, f"state: {drowzy_state}", (0, 250),
                                    cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.imshow(f"EyeBlink : ", image)
            if (cv2.waitKey(10) & 0xFF == ord('q')):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
