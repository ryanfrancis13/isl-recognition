import os
from pathlib import Path
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GLOG_minloglevel"] = "2"
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

SEQ_LEN = 30
POSE_IDX = [0, 2, 5, 9, 10, 11, 12, 13, 14, 15, 16]
FRAME_DIM = 63 + 63 + 6 + 2 + len(POSE_IDX) * 2  # 156
FEAT_DIM = FRAME_DIM + 6                         # 162 (adds 6 wrist-velocity features)

PILOT_SIGNS = ["IDLE", "HELLO", "THANK_YOU", "A"]
SIGNS = {
    "IDLE": "stream",
    "HELLO": "reps", "THANK_YOU": "reps", "YES": "reps", "NO": "reps",
    "PLEASE": "reps", "SORRY": "reps", "HELP": "reps",
    "GOOD_MORNING": "reps", "GOOD_NIGHT": "reps", "WATER": "reps",
    **{str(d): "stream" for d in range(10)},
    **{c: ("reps" if c in ("H", "J", "Y") else "stream") for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
}

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def make_holistic():
    return mp_holistic.Holistic(
        model_complexity=1, min_detection_confidence=0.6, min_tracking_confidence=0.6
    )


def _lm_xyz(lm_list):
    if lm_list is None:
        return None
    return np.array([[p.x, p.y, p.z] for p in lm_list.landmark], dtype=np.float32)


def _norm_hand(pts):
    if pts is None:
        return np.zeros(63, dtype=np.float32), False
    rel = pts - pts[0]
    scale = float(np.linalg.norm(rel[9]))
    if scale > 1e-6:
        rel /= scale
    return rel.flatten(), True


def frame_features(pose_pts, left_pts, right_pts):
    r_shape, r_ok = _norm_hand(right_pts)
    l_shape, l_ok = _norm_hand(left_pts)

    if pose_pts is not None:
        center = 0.5 * (pose_pts[11] + pose_pts[12])
        width = max(float(np.linalg.norm((pose_pts[11] - pose_pts[12])[:2])), 1e-3)
        upper = ((pose_pts[POSE_IDX, :2] - center[:2]) / width).flatten()
    else:
        center, width = np.array([0.5, 0.5, 0.0], dtype=np.float32), 1.0
        upper = np.zeros(len(POSE_IDX) * 2, dtype=np.float32)

    r_wrist = (right_pts[0] - center) / width if r_ok else np.zeros(3, dtype=np.float32)
    l_wrist = (left_pts[0] - center) / width if l_ok else np.zeros(3, dtype=np.float32)
    flags = np.array([float(r_ok), float(l_ok)], dtype=np.float32)
    return np.concatenate([r_shape, l_shape, r_wrist, l_wrist, flags, upper]).astype(np.float32)


def process_frame(holistic, frame):
    # MediaPipe needs the unflipped frame so anatomical left/right hands aren't swapped
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = holistic.process(rgb)
    feat = frame_features(
        _lm_xyz(res.pose_landmarks),
        _lm_xyz(res.left_hand_landmarks),
        _lm_xyz(res.right_hand_landmarks),
    )
    return feat, res


def to_window(frames, target_len=SEQ_LEN):
    n = len(frames)
    if n != target_len:
        old_t = np.linspace(0.0, 1.0, n)
        new_t = np.linspace(0.0, 1.0, target_len)
        frames = np.column_stack([np.interp(new_t, old_t, frames[:, c]) for c in range(frames.shape[1])]).astype(np.float32)
    wrists = frames[:, 126:132]
    vel = np.vstack([np.zeros((1, 6), dtype=np.float32), np.diff(wrists, axis=0)])
    return np.hstack([frames, vel])


def draw_skeleton(frame, res):
    if res.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
        )
    for hand in (res.left_hand_landmarks, res.right_hand_landmarks):
        if hand:
            mp_drawing.draw_landmarks(frame, hand, mp_holistic.HAND_CONNECTIONS)

