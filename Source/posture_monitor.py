import cv2, mediapipe as mp, numpy as np, time, platform
from collections import deque

# ============================================
# Desk Posture Monitor — readable, compact notes
# ============================================

# =========================
# ----- App Settings -----
# =========================

EMA_ALPHA = 0.3                  # float: generic EMA smoothing factor
CALIBRATION_SECONDS = 6          # int: stop calibration after this many seconds
CALIBRATION_FRAMES = 60          # int: or after this many frames (whichever hits first)

SHOW_SETUP_INSTRUCTIONS = True   # bool: show the startup instructions window
TARGET_WIDTH = 1280              # int: camera capture width (px)
TARGET_HEIGHT = 720              # int: camera capture height (px)
TARGET_FPS = 30                  # int: requested camera FPS
SHOW_FPS = True                  # bool: draw FPS readout

PANEL_ALPHA = 0.40               # float: translucent panel opacity
FONT = cv2.FONT_HERSHEY_SIMPLEX  # OpenCV font handle
COLOR_GOOD = (80, 220, 90)       # BGR: green-ish for “good”
COLOR_BAD = (0, 0, 255)          # BGR: red for “needs attention”
COLOR_WARN = (0, 180, 240)       # BGR: cyan/orange-ish for tips
COLOR_INFO = (235, 235, 235)     # BGR: light grey text
COLOR_GUIDE = (110, 180, 255)    # BGR: guide lines color
PANEL_BG = (0, 0, 0)             # BGR: panel background

mp_pose = mp.solutions.pose      # module: MediaPipe pose
MIN_VIS = 0.5                    # float: minimum landmark visibility to trust

# =========================
# ----- Trunk Classifier (slightly lenient) -----
# =========================

ENTER_Z = 1.30                   # float: score to enter "SLOUCHING"
EXIT_Z = 0.90                    # float: score to exit back to "GOOD"
FRAMES_ENTER = 8                 # int: hysteresis — frames needed to go bad
FRAMES_EXIT = 2                  # int: hysteresis — frames needed to recover
VIS_GATE = 0.50                  # float: mean visibility gate to adapt baselines
MEDIAN_WIN = 5                   # int: rolling median window for trunk features

ADAPT_ALPHA = 0.02               # float: adaptation speed while GOOD
SIGMA_FLOOR_TFA = 1.0            # float: min std for trunk-forward-angle
SIGMA_FLOOR_RFH = 0.02           # float: min std for relative face height
SIGMA_FLOOR_HDT = 2.0            # float: min std for head tilt
SIGMA_FLOOR_NSS = 1.0            # float: min std for neck slope

MARGIN_Z = 0.15                  # float: general z-margin
TFA_MARGIN_Z = 0.10              # float: margin for TFA
NSS_MARGIN_Z = 0.10              # float: margin for NSS

TFA_FAST_DEG = 4.0               # float: “fast” TFA jump (deg) — trip sooner
TFA_MIN_DEG = 3.0                # float: borderline TFA bump
NSS_MIN_DEG = 5.0                # float: borderline neck-slope bump
COMBO_MIN_DEG = 8.0              # float: borderline combined TFA+NSS

# ----- Shoulder-drop penalty (detect sinking shoulders) -----
SHDROP_MARGIN = 0.025            # float: ignore tiny trunk-length changes
SHDROP_GAIN   = 8.0              # float: scale for the drop penalty
SHDROP_QUICK  = 0.10             # float: % drop that triggers a quick slouch

# ----- Stable anchors (neck/hip EMA) -----
NECK_EMA_ALPHA = 0.35            # float: EMA smoothing for neck anchor
HIP_EMA_ALPHA  = 0.35            # float: EMA smoothing for hip anchor
ANCHOR_HOLD_FRAMES = 24          # int: keep last good anchor this many frames

# =========================
# ----- Wrist / Typing Risk -----
# =========================

EMA_ALPHA_WRIST = 0.25           # float: wrist angle EMA smoothing
MEDIAN_WIN_WRIST = 5             # int: rolling median window for wrists
SIGMA_FLOOR_WXT = 4.0            # float: min std for wrist extension z-score
SIGMA_FLOOR_WDEV = 4.0           # float: min std for wrist deviation z-score

WXT_SOFT = 24.0                  # float: soft wrist extension threshold (deg)
WXT_HARD = 27.0                  # float: hard wrist extension threshold (deg)
WDEV_SOFT = 16.0                 # float: soft wrist deviation threshold (deg)
WDEV_HARD = 18.0                 # float: hard wrist deviation threshold (deg)

WR_ENTER_Z = 1.25                # float: wrist risk z to enter AT_RISK
WR_EXIT_Z  = 0.85                # float: wrist risk z to exit to OK
WR_FRAMES_ENTER = 10             # int: frames to trip wrist risk
WR_FRAMES_EXIT  = 2              # int: frames to clear wrist risk

GRACE_EXT = 8.0                  # float: extension grace around baseline (deg)
GRACE_DEV = 7.0                  # float: deviation grace around baseline (deg)

TYPING_STD_MIN = 1.0             # float: min Y-std to count as typing motion
TYPING_STD_MAX = 20.0            # float: max Y-std to count as typing motion
TYPING_WIN = 45                  # int: samples kept for typing std
TYPING_RECENT_SEC = 2.0          # float: keep “typing context” alive this long

# ----- Keyboard contact detection -----
CONTACT_NEAR_NORM = 0.07         # float: hand Y must be within this (in trunk_len units)
CONTACT_STILL_STD = 1.0          # float: std of hand Y to count as resting
CONTACT_STILL_FRAMES = 60        # int: frames needed to confirm stillness

# ----- Elbow height vs keyboard plane -----
ELBOW_LOW_NORM  = 0.29           # float: “too low” elbow threshold (normalized)
ELBOW_HIGH_NORM = -0.20          # float: “too high” elbow threshold (normalized)
ELBOW_FAR_BIAS_LOW  = +0.05      # float: leniency for far arm (low)
ELBOW_FAR_BIAS_HIGH = +0.05      # float: leniency for far arm (high)
ELBOW_HYST = 0.05                # float: elbow hysteresis
ELBOW_DEADBAND_NORM = 0.07       # float: deadband around thresholds
HAND_NEAR_PLANE_NORM = 0.03      # float: if hand near plane, don’t flag high elbow
RIGHT_ELBOW_LOW_DELTA = -0.02    # float: slight tweak for right elbow low

# ----- Upper-arm flare (abduction) -----
FLARE_SOFT_DEG = 22.0            # float: soft flare threshold (deg)
FLARE_HARD_DEG = 29.0            # float: hard flare threshold (deg)
FLARE_EXIT_HYST = 6.0            # float: exit hysteresis (deg)
FLARE_DEADBAND = 4.0             # float: deadband (deg)
RIGHT_FLARE_TWEAK_SOFT = -3.0    # float: soft tweak for right arm (deg)
RIGHT_FLARE_TWEAK_HARD = -6.0    # float: hard tweak for right arm (deg)

# ----- Forearm incline (typing angle) -----
FOREARM_INCLINE_SOFT_DEG = 11.0  # float: soft incline threshold (deg)
FOREARM_INCLINE_HARD_DEG = 16.0  # float: hard incline threshold (deg)
RIGHT_FOREARM_TWEAK = -2.0       # float: tweak for right arm incline (deg)

# ----- Visibility / occlusion handling -----
VIS_ARM_GATE = 0.55              # float: min arm visibility to judge it
ELBOW_HOLD_FRAMES = 28           # int: keep last good elbow when occluded
FAR_VIS_CLEAR = 0.80             # float: far arm treated as “clear” above this

# ----- Hand anchor smoothing -----
HAND_EMA_ALPHA = 0.35            # float: EMA for hand anchor
HAND_STICK_FRAMES = 12           # int: keep last anchor this many frames

# ----- Protect plane adaptation from noise -----
PLANE_ADAPT_FRAMES = 12          # int: samples needed before plane nudges

# =========================
# ----- Utils -----
# =========================

def get_platform_backends():
    s = platform.system()  # str: OS name
    if s == "Windows": return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if s == "Linux":   return [cv2.CAP_V4L2, cv2.CAP_ANY]
    if s == "Darwin":  return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_ANY]

def find_cameras(max_index=6, backends=None):
    backends = backends or get_platform_backends()  # list[int]: CAP backends
    avail = []                                      # list[int]: found indices
    for i in range(max_index):                      # i: camera index
        for b in backends:                          # b: CAP backend enum
            cap = cv2.VideoCapture(i, b)            # cap: candidate capture
            if cap.isOpened():
                avail.append(i); cap.release(); break
            cap.release()
    return sorted(set(avail))

def open_camera(index, width=TARGET_WIDTH, height=TARGET_HEIGHT, fps=TARGET_FPS):
    # index: int camera index; width/height/fps: requested capture settings
    cap = None                                      # cv2.VideoCapture|None
    for b in get_platform_backends():
        t = cv2.VideoCapture(index, b)              # t: candidate capture
        if t.isOpened(): cap = t; break
    if cap is None or not cap.isOpened(): return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    return cap

def angle_between_vectors(v1, v2):
    # v1, v2: iterable[2]; return angle (deg) or None for degenerate vectors
    v1 = np.array(v1, np.float32); v2 = np.array(v2, np.float32)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return None
    u1 = v1/n1; u2 = v2/n2
    dot = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))

def ema(new_val, prev, alpha=EMA_ALPHA):
    # simple EMA with None-handling
    if new_val is None: return prev
    if prev is None:    return new_val
    return alpha*new_val + (1-alpha)*prev

def vis_ok(lm, thr=MIN_VIS):
    # lm: landmark; returns True if visibility >= thr
    try: return (lm.visibility or 0.0) >= thr
    except: return False

def draw_panel(frame, tl, br, color=PANEL_BG, alpha=PANEL_ALPHA):
    # frame: BGR image; tl/br: (x,y); draw a translucent panel
    x1, y1 = tl; x2, y2 = br
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

def safe_fmt(v, u=""):
    # best-effort number format; handles None/NaN
    if v is None: return f"N/A {u}".rstrip()
    try:
        if isinstance(v, float) and (v != v):  # NaN
            return f"N/A {u}".rstrip()
        return f"{v:.2f} {u}".rstrip()
    except:
        return f"N/A {u}".rstrip()

def line_col(state):
    # quick color pick for OK/FIX
    return (80, 220, 90) if state == "OK" else (0, 0, 255)

def vis2px(pt, w, h):
    # convert mediapipe normalized pt -> pixel coords
    return np.array([pt.x*w, pt.y*h], np.float32)

def first_non_none(*vals):
    # return first non-None value
    for v in vals:
        if v is not None: return v
    return None

# =========================
# ----- Coaching line -----
# =========================

def simple_suggestion(trunk_bad, flare_hard_any, elbow_low_any, elbow_high_any,
                      forearm_bad_any, wr_bad_any, contact_any):
    # pick a one-liner tip based on top issue
    if trunk_bad:       return "Sit tall"
    if elbow_low_any:   return "Raise elbows"
    if elbow_high_any:  return "Lower elbows"
    if flare_hard_any:  return "Bring elbows in"
    if forearm_bad_any: return "Keep forearms level"
    if wr_bad_any:      return "Straighten wrist"
    if contact_any:     return "Float wrist"
    return "Looking good"

# =========================
# ----- Setup screen -----
# =========================

def display_setup_window():
    # draw the startup instructions window and wait for SPACE / 'q'
    txt = [
        "DESK POSTURE MONITOR","",
        "1) Camera at desk height, clean side view.",
        "2) Keep head, shoulders, elbows, both arms, and hips in frame.",
        "3) 6-sec calibration: sit upright in ergonomically optimal posture;",
        "   type normally to set the keyboard plane.",
        "4) During calibration, overlays are hidden; just follow the prompt.",
        "",
        "Keys:  SPACE - start   r - recalibrate    a - advanced stats   q quit",
    ]
    w, h = 980, 540                                     # int,int: window size
    window = np.ones((h, w, 3), np.uint8) * 18          # np.ndarray: dark bg
    y = 72                                              # int: text y cursor

    for i, line in enumerate(txt):
        if i == 0:
            sz, col, th = 1.05, (80, 220, 90), 3        # title
        elif line == "":
            y += 6; continue
        elif line.startswith("Keys"):
            sz, col, th = 0.9, (235, 235, 235), 2       # footer keys
        else:
            sz, col, th = 0.78, (235, 235, 235), 2      # body
        cv2.putText(window, line, (44, y), FONT, sz, col, th, cv2.LINE_AA)
        y += int(34 if sz > 1.0 else 30)

    cv2.namedWindow("Setup", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Setup", w, h)
    cv2.imshow("Setup", window)

    while True:
        k = cv2.waitKey(30) & 0xFF                      # int: keycode
        if k == ord('q'):
            cv2.destroyWindow("Setup"); raise SystemExit
        if k == 32:  # SPACE
            break
    cv2.destroyWindow("Setup")

# =========================
# ----- Main loop -----
# =========================

def main():
    if SHOW_SETUP_INSTRUCTIONS:
        display_setup_window()

    cams = find_cameras()                                # list[int]: camera IDs
    if not cams:
        print("No cameras detected."); return
    cam_index = cams[-1] if len(cams) > 1 else cams[0]  # int: prefer external
    print(f"Using camera index {cam_index}")

    cap = open_camera(cam_index)                         # cv2.VideoCapture|None
    if cap is None:
        print("Failed to open camera."); return

    cv2.namedWindow("Posture Monitoring", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Posture Monitoring", 1280, 896)

    pose = mp_pose.Pose(
        static_image_mode=False,      # bool: live tracking
        model_complexity=1,           # int: 0/1/2 — 1 is a good balance
        enable_segmentation=False,    # bool: not needed
        min_detection_confidence=0.5, # float: detector threshold
        min_tracking_confidence=0.5   # float: tracker threshold
    )

    with pose as pose:
        # -------- Calibration buffers (lists of floats) --------
        calib_TFA = []      # list[float]: trunk forward angle
        calib_RFH = []      # list[float]: relative face height along trunk normal
        calib_HDT = []      # list[float]: head tilt
        calib_NSS = []      # list[float]: neck slope
        calib_TLEN = []     # list[float]: trunk length (px)

        # Baselines (medians) and running means/std
        base_TFA = base_RFH = base_HDT = base_NSS = None  # float|None
        base_TLEN = None                                   # float|None
        mu_TFA = sigma_TFA = mu_RFH = sigma_RFH = None     # float|None
        mu_HDT = sigma_HDT = mu_NSS = sigma_NSS = None     # float|None

        # Calibration state
        calibrating = True                # bool: currently calibrating
        calibration_start = time.time()   # float: start time (s)
        calibration_frames = 0            # int: frames seen during calibration

        # Trunk smoothing (EMA + median windows)
        sm_TFA = sm_RFH = sm_HDT = sm_NSS = None           # float|None
        hist_TFA = deque(maxlen=MEDIAN_WIN)                # deque[float]
        hist_RFH = deque(maxlen=MEDIAN_WIN)                # deque[float]
        hist_HDT = deque(maxlen=MEDIAN_WIN)                # deque[float]
        hist_NSS = deque(maxlen=MEDIAN_WIN)                # deque[float]

        # Stable anchors (neck/hip)
        neck_base_px = None          # np.ndarray([x,y])|None
        hip_base_px  = None          # np.ndarray([x,y])|None
        neck_miss = 0                # int: frames since last neck candidate
        hip_miss  = 0                # int: frames since last hip candidate

        # Timing / FPS
        last_t = time.time()         # float: last timestamp
        fps = 0.0                    # float: smoothed FPS
        fps_q = deque(maxlen=15)     # deque[float]: FPS smoothing window

        # Global trunk state
        state = "GOOD"               # str: "GOOD" | "SLOUCHING"
        enter_count = exit_count = 0 # int: hysteresis counters

        # Arm bookkeeping
        sides = ['L', 'R']           # list[str]: left / right arms

        # Wrist angle smoothing + history
        sm_WXT  = {s: None for s in sides}                              # float|None
        sm_WDEV = {s: None for s in sides}                              # float|None
        hist_WXT  = {s: deque(maxlen=MEDIAN_WIN_WRIST) for s in sides}  # deque[float]
        hist_WDEV = {s: deque(maxlen=MEDIAN_WIN_WRIST) for s in sides}  # deque[float]

        # Wrist calibration stats
        calib_WXT   = {s: [] for s in sides}  # dict[str,list[float]]
        calib_WDEV  = {s: [] for s in sides}  # dict[str,list[float]]
        mu_WXT      = {s: None for s in sides}        # dict[str,float|None]
        mu_WDEV     = {s: None for s in sides}        # dict[str,float|None]
        sigma_WXT   = {s: None for s in sides}        # dict[str,float|None]
        sigma_WDEV  = {s: None for s in sides}        # dict[str,float|None]
        base_WXT    = {s: None for s in sides}        # dict[str,float|None]
        base_WDEV   = {s: None for s in sides}        # dict[str,float|None]

        # Keyboard plane per arm
        calib_wristY      = {s: [] for s in sides}    # dict[str,list[float]] (px)
        keyboard_plane_y  = {s: None for s in sides}  # dict[str,float|None] (px)

        # Wrist risk FSM
        wr_state = {s: "OK" for s in sides}           # dict[str,str]: "OK"/"AT_RISK"
        wr_enter = {s: 0 for s in sides}              # dict[str,int]: enter counter
        wr_exit  = {s: 0 for s in sides}              # dict[str,int]: exit counter
        hard_WXT = {s: 0 for s in sides}              # dict[str,int]: hard ext streak
        hard_WDEV= {s: 0 for s in sides}              # dict[str,int]: hard dev streak

        # Typing stats
        wrY_hist = {s: deque(maxlen=TYPING_WIN) for s in sides}   # dict[str,deque]
        typing_stats = {s: {"stdY": 0.0, "typing": False} for s in sides}  # dict
        last_typing_time = {s: 0.0 for s in sides}                # dict[str,float]

        # Hand anchors
        sm_hand_px = {s: None for s in sides}   # dict[str,np.ndarray|None]
        hand_miss  = {s: 0 for s in sides}      # dict[str,int]

        # Elbow occlusion hold
        last_good_elbow  = {s: None for s in sides}  # dict[str,np.ndarray|None]
        elbow_miss_count = {s: 0 for s in sides}     # dict[str,int]

        # UI / session tracking
        advanced_mode = False            # bool: show advanced panel
        session_started = False          # bool: after calibration success
        session_start_time = None        # float|None: session start ts
        frames_total = 0                 # int: total frames since start
        frames_good  = 0                 # int: frames judged “good”

        while True:
            ok, frame = cap.read()                       # bool, np.ndarray(BGR)
            if not ok:
                print("Failed to grab frame"); break
            frame = cv2.resize(frame, (1280, 896))      # np.ndarray: working canvas
            h, w = frame.shape[:2]                      # int,int: frame dims

            # FPS calc
            now = time.time()                           # float: current ts
            inst_fps = 1.0 / max(1e-6, now - last_t)    # float: instantaneous FPS
            last_t = now
            fps_q.append(inst_fps)
            fps = sum(fps_q) / len(fps_q)               # float: smoothed FPS

            # Run MediaPipe
            frame.flags.writeable = False
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # np.ndarray(RGB)
            results = pose.process(rgb)                    # MediaPipe output
            frame.flags.writeable = True

            # Presence flag gates the HUD
            pose_present = (results.pose_landmarks is not None)  # bool

            # UI rollups
            contact_any = False          # bool: any wrist resting on plane
            flare_hard_any = False       # bool: any hard abduction
            elbow_low_any = False        # bool: any low elbow
            elbow_high_any = False       # bool: any high elbow
            wr_bad_any = False           # bool: any wrist at risk
            forearm_bad_any = False      # bool: any steep forearm

            if pose_present:
                lm = results.pose_landmarks.landmark  # list[landmark]

                # Landmarks (mediapipe indices)
                L_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]   # landmark
                R_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]  # landmark
                L_hip= lm[mp_pose.PoseLandmark.LEFT_HIP]        # landmark
                R_hip= lm[mp_pose.PoseLandmark.RIGHT_HIP]       # landmark
                L_ear= lm[mp_pose.PoseLandmark.LEFT_EAR]        # landmark
                R_ear= lm[mp_pose.PoseLandmark.RIGHT_EAR]       # landmark
                L_eye= lm[mp_pose.PoseLandmark.LEFT_EYE]        # landmark
                R_eye= lm[mp_pose.PoseLandmark.RIGHT_EYE]       # landmark
                L_el = lm[mp_pose.PoseLandmark.LEFT_ELBOW]      # landmark
                R_el = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]     # landmark
                L_wr = lm[mp_pose.PoseLandmark.LEFT_WRIST]      # landmark
                R_wr = lm[mp_pose.PoseLandmark.RIGHT_WRIST]     # landmark
                L_idx= lm[mp_pose.PoseLandmark.LEFT_INDEX]      # landmark
                R_idx= lm[mp_pose.PoseLandmark.RIGHT_INDEX]     # landmark
                L_pky= lm[mp_pose.PoseLandmark.LEFT_PINKY]      # landmark
                R_pky= lm[mp_pose.PoseLandmark.RIGHT_PINKY]     # landmark
                L_th = lm[mp_pose.PoseLandmark.LEFT_THUMB]      # landmark
                R_th = lm[mp_pose.PoseLandmark.RIGHT_THUMB]     # landmark

                def px_if_ok(l):
                    # if vis ok -> pixel coords; else None
                    return np.array([l.x*w, l.y*h], np.float32) if vis_ok(l) else None

                # pixel coords or None
                L_sh_px = px_if_ok(L_sh); R_sh_px = px_if_ok(R_sh)
                L_hip_px= px_if_ok(L_hip); R_hip_px= px_if_ok(R_hip)
                L_ear_px= px_if_ok(L_ear); R_ear_px= px_if_ok(R_ear)
                L_eye_px= px_if_ok(L_eye); R_eye_px= px_if_ok(R_eye)
                L_el_px = px_if_ok(L_el);  R_el_px = px_if_ok(R_el)
                L_wr_px = px_if_ok(L_wr);  R_wr_px = px_if_ok(R_wr)

                # ------ Stable neck & hip bases ------
                def avg_pts(*pts):
                    # average multiple [x,y] points ignoring Nones
                    arr = [p for p in pts if p is not None]
                    if not arr: return None
                    return np.mean(np.stack(arr, 0), axis=0).astype(np.float32)

                neck_candidate = avg_pts(L_sh_px, R_sh_px)   # np.ndarray|None
                hip_candidate  = avg_pts(L_hip_px, R_hip_px) # np.ndarray|None

                if neck_candidate is not None:
                    neck_base_px = neck_candidate if neck_base_px is None else \
                                   NECK_EMA_ALPHA*neck_candidate + (1-NECK_EMA_ALPHA)*neck_base_px
                    neck_miss = 0
                else:
                    neck_miss += 1
                    if neck_miss > ANCHOR_HOLD_FRAMES: neck_base_px = None

                if hip_candidate is not None:
                    hip_base_px = hip_candidate if hip_base_px is None else \
                                  HIP_EMA_ALPHA*hip_candidate + (1-HIP_EMA_ALPHA)*hip_base_px
                    hip_miss = 0
                else:
                    hip_miss += 1
                    if hip_miss > ANCHOR_HOLD_FRAMES: hip_base_px = None

                # Safe face centers (prefer single-sided if only one visible)
                ear_center = first_non_none(avg_pts(L_ear_px), avg_pts(R_ear_px),
                                            avg_pts(L_ear_px, R_ear_px))  # np.ndarray|None
                eye_center = first_non_none(avg_pts(L_eye_px), avg_pts(R_eye_px),
                                            avg_pts(L_eye_px, R_eye_px))   # np.ndarray|None

                # Hand anchor per arm using multiple fingertips when available
                def hand_anchor_for(S):
                    # S: 'L' | 'R' -> np.ndarray([x,y])|None
                    if S == "L":
                        c = []
                        if vis_ok(L_idx): c.append(vis2px(L_idx, w, h))
                        if vis_ok(L_pky): c.append(vis2px(L_pky, w, h))
                        if vis_ok(L_th ): c.append(vis2px(L_th , w, h))
                        if len(c) >= 2:   return np.median(np.stack(c, 0), 0).astype(np.float32)
                        if vis_ok(L_wr):  return vis2px(L_wr, w, h)
                    else:
                        c = []
                        if vis_ok(R_idx): c.append(vis2px(R_idx, w, h))
                        if vis_ok(R_pky): c.append(vis2px(R_pky, w, h))
                        if vis_ok(R_th ): c.append(vis2px(R_th , w, h))
                        if len(c) >= 2:   return np.median(np.stack(c, 0), 0).astype(np.float32)
                        if vis_ok(R_wr):  return vis2px(R_wr, w, h)
                    return None

                # keep hand anchors sticky through brief occlusion
                sm_hand_px = {'L': None, 'R': None} | sm_hand_px     # dict[str,np.ndarray|None]
                for s in sides:
                    a = hand_anchor_for(s)                            # np.ndarray|None
                    if a is not None:
                        sm_hand_px[s] = a if sm_hand_px[s] is None else HAND_EMA_ALPHA*a + (1-HAND_EMA_ALPHA)*sm_hand_px[s]
                        hand_miss[s] = 0
                    else:
                        hand_miss[s] += 1
                        if hand_miss[s] > HAND_STICK_FRAMES: sm_hand_px[s] = None

                # Typing detection per arm (std of hand Y)
                for s in sides:
                    if sm_hand_px[s] is not None:
                        wrY_hist[s].append(float(sm_hand_px[s][1]))
                    stdY = np.std(wrY_hist[s]) if len(wrY_hist[s]) >= 8 else 0.0  # float
                    typing = (TYPING_STD_MIN <= stdY <= TYPING_STD_MAX)            # bool
                    if typing: last_typing_time[s] = now
                    typing_stats[s]["stdY"] = stdY
                    typing_stats[s]["typing"] = typing

                # ---------- Trunk features (TFA/RFH/HDT/NSS) ----------
                if neck_base_px is None or hip_base_px is None or ear_center is None:
                    TFA = RFH = HDT = NSS = None
                    trunk_len = None
                else:
                    trunk_vec = neck_base_px - hip_base_px               # np.ndarray([dx,dy])
                    trunk_len = float(np.linalg.norm(trunk_vec))         # float: length
                    if trunk_len < 1e-6:
                        TFA = RFH = HDT = NSS = None
                    else:
                        global_vertical = np.array([0.0, -1.0], np.float32)              # np.ndarray
                        TFA_raw = angle_between_vectors(trunk_vec, global_vertical)      # float|None (deg)

                        trunk_unit = trunk_vec / trunk_len                               # np.ndarray([ux,uy])
                        trunk_normal = np.array([-trunk_unit[1], trunk_unit[0]], np.float32)  # np.ndarray

                        RFH_raw = float(np.dot((ear_center - neck_base_px), trunk_normal)) / max(trunk_len, 1e-6)  # float
                        NSS_raw = angle_between_vectors(ear_center - neck_base_px, global_vertical)                # float

                        # head tilt if eyes below ears AND horizontal angle present
                        HDT_raw = 0.0
                        if eye_center is not None:
                            v = eye_center - ear_center                                  # np.ndarray
                            ang_h = angle_between_vectors(v, np.array([1.0, 0.0], np.float32))
                            if eye_center[1] > ear_center[1] and ang_h is not None:
                                HDT_raw = ang_h

                        def smooth_update(raw, sm, hist):
                            # EMA + rolling median
                            if raw is None: return sm
                            sm = ema(raw, sm)
                            hist.append(sm if sm is not None else raw)
                            return float(np.median(hist)) if len(hist) else sm

                        TFA = smooth_update(TFA_raw, sm_TFA, hist_TFA); sm_TFA = TFA if TFA is not None else sm_TFA
                        RFH = smooth_update(RFH_raw, sm_RFH, hist_RFH); sm_RFH = RFH if RFH is not None else sm_RFH
                        HDT = smooth_update(HDT_raw, sm_HDT, hist_HDT); sm_HDT = HDT if HDT is not None else sm_HDT
                        NSS = smooth_update(NSS_raw, sm_NSS, hist_NSS); sm_NSS = NSS if NSS is not None else sm_NSS

                # ----- Wrist angle helpers -----
                def wrist_extension_deg(elp, wrp):
                    # angle of forearm vs horizontal (negative = extension up)
                    if elp is None or wrp is None: return None
                    dx = abs(wrp[0] - elp[0]); dy = (wrp[1] - elp[1])
                    if dx < 1e-3 and abs(dy) < 1e-3: return None
                    return float(-np.degrees(np.arctan2(dy, dx + 1e-6)))

                def wrist_deviation_deg(elp, wrp):
                    # deviation relative to forearm axis (radial/ulnar)
                    if elp is None or wrp is None: return None
                    fore = wrp - elp; n = np.linalg.norm(fore)
                    if n < 1e-6: return None
                    u = fore / n; perp = np.array([-u[1], u[0]], np.float32)
                    dev_px = np.dot((wrp - elp), perp)
                    return float(np.degrees(np.arctan2(dev_px, n)))

                # Raw wrist angles per side
                WXT_raw = {
                    'L': wrist_extension_deg(L_el_px, L_wr_px) if (L_el_px is not None and L_wr_px is not None) else None,
                    'R': wrist_extension_deg(R_el_px, R_wr_px) if (R_el_px is not None and R_wr_px is not None) else None
                }
                WDEV_raw = {
                    'L': wrist_deviation_deg(L_el_px, L_wr_px) if (L_el_px is not None and L_wr_px is not None) else None,
                    'R': wrist_deviation_deg(R_el_px, R_wr_px) if (R_el_px is not None and R_wr_px is not None) else None
                }

                def smooth_wrist(raw, sm, hist):
                    # EMA + rolling median for wrist angles
                    if raw is None: return sm
                    sm = EMA_ALPHA_WRIST*raw + (1-EMA_ALPHA_WRIST)*(sm if sm is not None else raw)
                    hist.append(sm)
                    return float(np.median(hist)) if len(hist) else sm

                for s in sides:
                    sm_WXT[s]  = smooth_wrist(WXT_raw[s],  sm_WXT[s],  hist_WXT[s])
                    sm_WDEV[s] = smooth_wrist(WDEV_raw[s], sm_WDEV[s], hist_WDEV[s])

                # -------- Calibration logic --------
                if calibrating:
                    elapsed = time.time() - calibration_start   # float: secs since start
                    calibration_frames += 1                     # int: frames seen

                    if TFA is not None: calib_TFA.append(TFA)
                    if RFH is not None: calib_RFH.append(RFH)
                    if HDT is not None: calib_HDT.append(HDT)
                    if NSS is not None: calib_NSS.append(NSS)
                    if 'trunk_len' in locals() and trunk_len is not None: calib_TLEN.append(trunk_len)

                    for s in sides:
                        if sm_hand_px[s] is not None: calib_wristY[s].append(float(sm_hand_px[s][1]))
                        if sm_WXT[s]  is not None:    calib_WXT[s].append(sm_WXT[s])
                        if sm_WDEV[s] is not None:    calib_WDEV[s].append(sm_WDEV[s])

                    # draw calibration panel
                    draw_panel(frame, (24, 22), (1240, 220))
                    cv2.putText(frame, "Status: CALIBRATING", (46, 66), FONT, 1.1, COLOR_GOOD, 4, cv2.LINE_AA)
                    cv2.putText(frame, "Tip: Type normally to set the keyboard plane.", (46, 110), FONT, 0.80, COLOR_INFO, 2, cv2.LINE_AA)
                    cv2.putText(frame,
                                f"Time left: {max(0, int(CALIBRATION_SECONDS - elapsed))}s   |   Frames: {calibration_frames}",
                                (46, 148), FONT, 0.70, COLOR_INFO, 2, cv2.LINE_AA)

                    # finalize calibration when enough data collected
                    if calibration_frames >= CALIBRATION_FRAMES or elapsed >= CALIBRATION_SECONDS:
                        ok_sh = (len(calib_TFA) > 5 and len(calib_RFH) > 5 and
                                 len(calib_HDT) > 5 and len(calib_NSS) > 5 and len(calib_TLEN) > 5)
                        if ok_sh:
                            # trunk stats
                            mu_TFA   = float(np.mean(calib_TFA)); sigma_TFA = float(np.std(calib_TFA, ddof=1)) or 1e-3
                            mu_RFH   = float(np.mean(calib_RFH)); sigma_RFH = float(np.std(calib_RFH, ddof=1)) or 1e-3
                            mu_HDT   = float(np.mean(calib_HDT)); sigma_HDT = float(np.std(calib_HDT, ddof=1)) or 1e-3
                            mu_NSS   = float(np.mean(calib_NSS)); sigma_NSS = float(np.std(calib_NSS, ddof=1)) or 1e-3
                            base_TFA = float(np.median(calib_TFA))
                            base_RFH = float(np.median(calib_RFH))
                            base_HDT = float(np.median(calib_HDT))
                            base_NSS = float(np.median(calib_NSS))
                            base_TLEN= float(np.median(calib_TLEN))

                            def finalize(arr, floor):
                                # produce (mu, sigma, base) with sigma floor
                                if len(arr) < 8: return (0.0, floor, 0.0)
                                mu = float(np.mean(arr))
                                sigma = float(np.std(arr, ddof=1)) or 1e-3
                                base = float(np.median(arr))
                                return mu, sigma, base

                            # wrists + keyboard plane
                            for s in sides:
                                mu_WXT[s],  sigma_WXT[s],  base_WXT[s]  = finalize(calib_WXT[s],  SIGMA_FLOOR_WXT)
                                mu_WDEV[s], sigma_WDEV[s], base_WDEV[s] = finalize(calib_WDEV[s], SIGMA_FLOOR_WDEV)

                                ys = np.array(calib_wristY[s], dtype=np.float32)  # np.ndarray
                                if ys.size >= 8:
                                    lo, hi = np.percentile(ys, 20), np.percentile(ys, 80)
                                    ys = ys[(ys >= lo) & (ys <= hi)]
                                    keyboard_plane_y[s] = float(np.median(ys if ys.size else
                                                                          np.array(calib_wristY[s], np.float32)))

                            # flip into live mode
                            calibrating = False
                            state = "GOOD"; enter_count = exit_count = 0
                            session_started = True
                            session_start_time = time.time()
                            frames_total = frames_good = 0
                            print("Calibration complete.")
                        else:
                            # not enough clean data — try again
                            calibration_start = time.time(); calibration_frames = 0
                            calib_TFA.clear(); calib_RFH.clear(); calib_HDT.clear(); calib_NSS.clear(); calib_TLEN.clear()
                            for s in sides: calib_WXT[s].clear(); calib_WDEV[s].clear(); calib_wristY[s].clear()

                    # footer keys + FPS
                    cv2.putText(frame, "r: recalibrate   a: advanced   q: quit",
                                (36, h-24), FONT, 0.65, (230, 230, 230), 2, cv2.LINE_AA)
                    if SHOW_FPS:
                        cv2.putText(frame, f"FPS: {fps:.1f}", (w-180, 40), FONT, 0.62, (210, 210, 210), 2, cv2.LINE_AA)
                    cv2.imshow("Posture Monitoring", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'): break
                    elif key == ord('r'):
                        # full reset during calibration
                        calibrating = True; calibration_start = time.time(); calibration_frames = 0
                        calib_TFA.clear(); calib_RFH.clear(); calib_HDT.clear(); calib_NSS.clear(); calib_TLEN.clear()
                        base_TFA = base_RFH = base_HDT = base_NSS = None; base_TLEN = None
                        mu_TFA = sigma_TFA = mu_RFH = sigma_RFH = None
                        mu_HDT = sigma_HDT = mu_NSS = sigma_NSS = None
                        sm_TFA = sm_RFH = sm_HDT = sm_NSS = None
                        hist_TFA.clear(); hist_RFH.clear(); hist_HDT.clear(); hist_NSS.clear()
                        state = "GOOD"; enter_count = exit_count = 0
                        neck_base_px = None; hip_base_px = None; neck_miss = hip_miss = 0
                        for s in sides:
                            calib_WXT[s].clear(); calib_WDEV[s].clear(); calib_wristY[s].clear()
                            mu_WXT[s] = mu_WDEV[s] = None; sigma_WXT[s] = sigma_WDEV[s] = None
                            base_WXT[s] = base_WDEV[s] = None
                            sm_WXT[s] = sm_WDEV[s] = None
                            hist_WXT[s].clear(); hist_WDEV[s].clear()
                            wr_state[s] = "OK"; wr_enter[s] = wr_exit[s] = 0
                            hard_WXT[s] = hard_WDEV[s] = 0
                            wrY_hist[s].clear()
                            typing_stats[s]["stdY"] = 0.0; typing_stats[s]["typing"] = False; last_typing_time[s] = 0.0
                            sm_hand_px[s] = None; hand_miss[s] = 0
                            last_good_elbow[s] = None; elbow_miss_count[s] = 0
                            keyboard_plane_y[s] = None
                        session_started = False; session_start_time = None; frames_total = frames_good = 0
                    elif key in (ord('a'), ord('A')):
                        advanced_mode = not advanced_mode
                    continue  # don’t draw overlays during calibration

                # ---------- (post-cal) trunk adaptation ----------
                used_vis = []
                for lmpt in (L_sh, R_sh, L_hip, R_hip, L_ear, R_ear, L_eye, R_eye, L_el, R_el, L_wr, R_wr):
                    if hasattr(lmpt, "visibility"):
                        used_vis.append(lmpt.visibility or 0.0)
                mean_vis = np.mean(used_vis) if used_vis else 0.0   # float

                if state == "GOOD" and mean_vis >= VIS_GATE and None not in (TFA, RFH, HDT, NSS):
                    def adapt(mu, sigma, val, floor):
                        mu = (1-ADAPT_ALPHA)*mu + ADAPT_ALPHA*val
                        dev = abs(val - mu)
                        sigma = (1-ADAPT_ALPHA)*sigma + ADAPT_ALPHA*max(dev, floor)
                        return mu, sigma

                    mu_TFA, sigma_TFA = adapt(mu_TFA, sigma_TFA, TFA, SIGMA_FLOOR_TFA)
                    mu_RFH, sigma_RFH = adapt(mu_RFH, sigma_RFH, RFH, SIGMA_FLOOR_RFH)
                    mu_HDT, sigma_HDT = adapt(mu_HDT, sigma_HDT, HDT, SIGMA_FLOOR_HDT)
                    mu_NSS, sigma_NSS = adapt(mu_NSS, sigma_NSS, NSS, SIGMA_FLOOR_NSS)

                    # wrist baselines adapt only while actively typing
                    def adapt_w(mu, sigma, val, floor):
                        mu = (1-ADAPT_ALPHA)*mu + ADAPT_ALPHA*val
                        dev = abs(val - mu)
                        sigma = (1-ADAPT_ALPHA)*sigma + ADAPT_ALPHA*max(dev, floor)
                        return mu, sigma

                    for s in sides:
                        if sm_WXT[s] is not None and typing_stats[s]["typing"]:
                            mu_WXT[s], sigma_WXT[s] = adapt_w(mu_WXT[s], sigma_WXT[s], sm_WXT[s], SIGMA_FLOOR_WXT)
                        if sm_WDEV[s] is not None and typing_stats[s]["typing"]:
                            mu_WDEV[s], sigma_WDEV[s] = adapt_w(mu_WDEV[s], sigma_WDEV[s], sm_WDEV[s], SIGMA_FLOOR_WDEV)

                        # gently nudge the keyboard plane toward current hand height during sustained typing
                        if typing_stats[s]["typing"] and sm_hand_px[s] is not None and keyboard_plane_y[s] is not None and 'trunk_len' in locals() and trunk_len:
                            if len(wrY_hist[s]) >= PLANE_ADAPT_FRAMES and np.std(list(wrY_hist[s])[-PLANE_ADAPT_FRAMES:]) >= TYPING_STD_MIN:
                                target = float(sm_hand_px[s][1])        # float: target plane (px)
                                max_step = 0.01 * trunk_len             # float: limit move per frame
                                delta = float(np.clip(target - keyboard_plane_y[s], -max_step, max_step))
                                keyboard_plane_y[s] += delta

                # ---------- Shoulder/neck classifier (+ shoulder drop) ----------
                if None not in (TFA, RFH, HDT, NSS):
                    def z_pos_only(val, mu, sigma, floor, margin):
                        s = max(sigma, floor)
                        z = max(0.0, (val - mu) / s)
                        return max(0.0, z - margin)

                    z_tfa = z_pos_only(TFA, mu_TFA, sigma_TFA, SIGMA_FLOOR_TFA, TFA_MARGIN_Z)
                    z_nss = z_pos_only(NSS, mu_NSS, sigma_NSS, SIGMA_FLOOR_NSS, NSS_MARGIN_Z)
                    z_rfh = z_pos_only(RFH, mu_RFH, sigma_RFH, SIGMA_FLOOR_RFH, MARGIN_Z)
                    z_hdt = z_pos_only(HDT, mu_HDT, sigma_HDT, MARGIN_Z, MARGIN_Z)

                    drop_ratio = None
                    if base_TLEN and trunk_len:
                        drop_ratio = max(0.0, 1.0 - (trunk_len / max(1e-6, base_TLEN)))  # float: 0..1
                    z_shdrop = SHDROP_GAIN * max(0.0, (drop_ratio or 0.0) - SHDROP_MARGIN)  # float

                    S = 0.30*z_rfh + 0.35*z_nss + 0.25*z_tfa + 0.10*z_hdt + z_shdrop       # float: score

                    d_tfa = TFA - mu_TFA                 # float: TFA delta vs mean
                    d_nss = NSS - mu_NSS                 # float: NSS delta vs mean
                    tfa_fast = (d_tfa >= TFA_FAST_DEG)   # bool: quick bump
                    borderline = ((d_tfa >= TFA_MIN_DEG) or (d_nss >= NSS_MIN_DEG) or
                                  ((d_tfa + d_nss) >= COMBO_MIN_DEG))                       # bool
                    shoulder_quick = (drop_ratio is not None and drop_ratio >= SHDROP_QUICK) # bool

                    if state == "GOOD":
                        if S >= ENTER_Z or tfa_fast or borderline or shoulder_quick:
                            # heavier increment when clear issues appear
                            enter_count += 2 if (tfa_fast or borderline or shoulder_quick) else 1
                        else:
                            enter_count = 0
                        if enter_count >= FRAMES_ENTER:
                            state = "SLOUCHING"; enter_count = 0; exit_count = 0
                    else:
                        close_all = (z_rfh < 0.7 and z_nss < 0.7 and z_hdt < 0.7 and z_tfa < 0.7 and z_shdrop < 0.5)
                        if S < EXIT_Z or close_all:
                            exit_count += 1 + (1 if close_all else 0)
                        else:
                            exit_count = 0
                        if exit_count >= FRAMES_EXIT:
                            state = "GOOD"; enter_count = exit_count = 0

                # ---------- Overlays ----------
                if L_sh_px is not None and R_sh_px is not None:
                    cv2.line(frame, (int(L_sh_px[0]), int(L_sh_px[1])),
                                   (int(R_sh_px[0]), int(R_sh_px[1])), (128, 210, 250), 2, cv2.LINE_AA)
                if neck_base_px is not None and ear_center is not None:
                    ui_color = COLOR_BAD if state == "SLOUCHING" else COLOR_GOOD
                    cv2.line(frame, (int(neck_base_px[0]), int(neck_base_px[1])),
                                   (int(ear_center[0]),   int(ear_center[1])), ui_color, 3, cv2.LINE_AA)
                    cv2.circle(frame, (int(ear_center[0]), int(ear_center[1])), 8, (240, 255, 80), -1, cv2.LINE_AA)
                if neck_base_px is not None:
                    cv2.circle(frame, (int(neck_base_px[0]), int(neck_base_px[1])), 8, (60, 230, 245), -1, cv2.LINE_AA)

                planes = [y for y in keyboard_plane_y.values() if y is not None]  # list[float]
                if planes:
                    plane_draw = float(np.median(np.array(planes, np.float32)))     # float: y px
                    cv2.line(frame, (24, int(plane_draw)), (int(w-24), int(plane_draw)), (150, 150, 150), 1, cv2.LINE_AA)

                shoulder_span = None                      # float|None: pixel span
                shoulder_axis_u = None                    # np.ndarray|None: unit axis
                if L_sh_px is not None and R_sh_px is not None:
                    shoulder_span = float(np.linalg.norm(R_sh_px - L_sh_px))
                    if shoulder_span and shoulder_span > 1e-6:
                        shoulder_axis_u = (R_sh_px - L_sh_px) / shoulder_span

                near_arm = 'L' if (vis_ok(L_sh) and vis_ok(R_sh) and L_sh.z < R_sh.z) else 'R'  # str: 'L'|'R'
                far_arm  = 'R' if near_arm == 'L' else 'L'                                     # str: 'L'|'R'

                def points_for(s):
                    # s: 'L'|'R' -> (shoulder_px, elbow_px, wrist_px)
                    return (L_sh_px, L_el_px, L_wr_px) if s == 'L' else (R_sh_px, R_el_px, R_wr_px)

                any_arm_visible = False     # bool: any arm judged this frame
                all_visible_arms_ok = True  # bool: every visible arm OK

                for s in sides:
                    sh_s, el_s_raw, wr_s = points_for(s)                    # np.ndarray|None
                    trip = ((L_sh, L_el, L_wr) if s == 'L' else (R_sh, R_el, R_wr))  # tuple[landmark]
                    v_list = [(lmpt.visibility or 0.0) for lmpt in trip]     # list[float]
                    arm_vis = float(np.mean(v_list))                         # float: mean vis

                    # elbow “hold” through brief occlusion
                    if el_s_raw is not None and (trip[1].visibility or 0.0) >= MIN_VIS:
                        el_s = el_s_raw.copy(); last_good_elbow[s] = el_s.copy(); elbow_miss_count[s] = 0
                    else:
                        elbow_miss_count[s] += 1
                        el_s = last_good_elbow[s].copy() if (last_good_elbow[s] is not None and
                                                           elbow_miss_count[s] <= ELBOW_HOLD_FRAMES) else None

                    arm_visible = (sh_s is not None) and (el_s is not None)  # bool
                    if not arm_visible:
                        continue
                    any_arm_visible = True

                    # trunk vector for local axes
                    if neck_base_px is not None and hip_base_px is not None:
                        tvec = neck_base_px - hip_base_px                     # np.ndarray
                    else:
                        if s == 'L' and (L_sh_px is not None and L_hip_px is not None):
                            tvec = L_sh_px - L_hip_px
                        elif s == 'R' and (R_sh_px is not None and R_hip_px is not None):
                            tvec = R_sh_px - R_hip_px
                        else:
                            continue
                    tlen = float(np.linalg.norm(tvec))                        # float
                    if tlen < 1e-6: continue
                    t_u = tvec / tlen                                         # np.ndarray: unit

                    # arm segment + flare angle
                    up_vec = el_s - sh_s                                      # np.ndarray
                    up_len = float(np.linalg.norm(up_vec))                    # float
                    if up_len < 1e-6: continue
                    up_u = up_vec / up_len                                    # np.ndarray
                    lat_u = (-shoulder_axis_u if s == 'L' else shoulder_axis_u) \
                             if shoulder_axis_u is not None else np.array([-t_u[1], t_u[0]], np.float32)
                    lat = max(0.0, float(np.dot(up_u, lat_u)))                # float
                    vert = abs(float(np.dot(up_u, t_u)))                      # float
                    flare_angle = float(np.degrees(np.arctan2(lat, max(1e-6, vert))))  # float deg

                    if not hasattr(main, "flare_soft_flag"): main.flare_soft_flag = {'L': False, 'R': False}
                    if not hasattr(main, "flare_hard_flag"): main.flare_hard_flag = {'L': False, 'R': False}
                    soft_flag = main.flare_soft_flag[s]                       # bool
                    hard_flag = main.flare_hard_flag[s]                       # bool

                    # far-arm biasing if visibility is poor
                    far_side = ('R' if vis_ok(L_sh) and vis_ok(R_sh) and L_sh.z < R_sh.z else 'L')  # str
                    far_bonus_soft = 0.0 if arm_vis >= FAR_VIS_CLEAR else (12.0 if s == far_side else 0.0)
                    far_bonus_hard = 0.0 if arm_vis >= FAR_VIS_CLEAR else (16.0 if s == far_side else 0.0)
                    soft_thr = FLARE_SOFT_DEG + far_bonus_soft               # float
                    hard_thr = FLARE_HARD_DEG + far_bonus_hard               # float
                    if s == 'R':
                        soft_thr += RIGHT_FLARE_TWEAK_SOFT
                        hard_thr += RIGHT_FLARE_TWEAK_HARD

                    # hysteresis around flare thresholds
                    if flare_angle >= (soft_thr + FLARE_DEADBAND): soft_flag = True
                    elif flare_angle <= (soft_thr - FLARE_EXIT_HYST): soft_flag = False
                    if flare_angle >= (hard_thr + FLARE_DEADBAND):  hard_flag = True
                    elif flare_angle <= (hard_thr - FLARE_EXIT_HYST): hard_flag = False
                    main.flare_soft_flag[s] = soft_flag
                    main.flare_hard_flag[s] = hard_flag

                    # forearm incline
                    fore_soft = False; fore_hard = False; fore_deg = None
                    if sm_hand_px[s] is not None:
                        fv = sm_hand_px[s] - el_s                               # np.ndarray
                        fore_deg = abs(float(np.degrees(np.arctan2(fv[1], fv[0]))))  # float deg
                        fsoft = FOREARM_INCLINE_SOFT_DEG + (RIGHT_FOREARM_TWEAK if s == 'R' else 0.0)
                        fhard = FOREARM_INCLINE_HARD_DEG + (RIGHT_FOREARM_TWEAK if s == 'R' else 0.0)
                        fore_soft = (fore_deg >= fsoft)
                        fore_hard = (fore_deg >= fhard)
                        if fore_hard: forearm_bad_any = True

                    stdY = typing_stats[s]["stdY"]                                # float
                    typing_active = typing_stats[s]["typing"]                      # bool
                    recent_typing = (now - last_typing_time[s]) <= TYPING_RECENT_SEC  # bool

                    # if not in typing context, draw minimal arm guides and skip risk checks
                    if not (typing_active or recent_typing):
                        up_col = (160, 210, 255)
                        cv2.line(frame, (int(sh_s[0]), int(sh_s[1])), (int(el_s[0]), int(el_s[1])), up_col, 3, cv2.LINE_AA)
                        if sm_hand_px[s] is not None:
                            col = line_col("OK")
                            cv2.line(frame, (int(el_s[0]), int(el_s[1])), (int(sm_hand_px[s][0]), int(sm_hand_px[s][1])), col, 2, cv2.LINE_AA)
                        continue

                    # keyboard contact (resting) detection
                    contact = False
                    if keyboard_plane_y[s] is not None and sm_hand_px[s] is not None and tlen:
                        near = abs(sm_hand_px[s][1] - keyboard_plane_y[s]) <= (CONTACT_NEAR_NORM * tlen)     # bool
                        still = (stdY <= CONTACT_STILL_STD and len(wrY_hist[s]) >= CONTACT_STILL_FRAMES)      # bool
                        contact = bool(near and still)                                                        # bool

                    # elbow height vs plane (normalized)
                    elbow_low = False; elbow_high = False; elbow_norm = None
                    if keyboard_plane_y[s] is not None and arm_vis >= VIS_ARM_GATE and tlen:
                        use_far_bias = (arm_vis < FAR_VIS_CLEAR)                          # bool
                        low_thr  = ELBOW_LOW_NORM  + (ELBOW_FAR_BIAS_LOW  if (use_far_bias and s == far_side) else 0.0)   # float
                        high_thr = ELBOW_HIGH_NORM - (ELBOW_FAR_BIAS_HIGH if (use_far_bias and s == far_side) else 0.0)   # float
                        if s == 'R': low_thr += RIGHT_ELBOW_LOW_DELTA

                        elbow_norm = (float(el_s[1]) - keyboard_plane_y[s]) / tlen       # float

                        if not hasattr(main, "elbow_low_flag"):  main.elbow_low_flag  = {'L': False, 'R': False}
                        if not hasattr(main, "elbow_high_flag"): main.elbow_high_flag = {'L': False, 'R': False}
                        low_flag  = main.elbow_low_flag[s]                               # bool
                        high_flag = main.elbow_high_flag[s]                              # bool

                        low_enter  = low_thr  + ELBOW_DEADBAND_NORM                      # float
                        low_exit   = low_thr  - ELBOW_HYST                               # float
                        high_enter = high_thr - ELBOW_DEADBAND_NORM                      # float
                        high_exit  = high_thr + ELBOW_HYST                               # float

                        hand_norm = None
                        if sm_hand_px[s] is not None:
                            hand_norm = (float(sm_hand_px[s][1]) - keyboard_plane_y[s]) / tlen  # float

                        if elbow_norm > low_enter:      low_flag = True
                        elif elbow_norm < low_exit:     low_flag = False

                        high_candidate = (elbow_norm < high_enter)
                        if hand_norm is not None and abs(hand_norm) <= HAND_NEAR_PLANE_NORM:
                            high_candidate = False
                        if high_candidate:              high_flag = True
                        elif elbow_norm > high_exit:    high_flag = False

                        elbow_low  = low_flag
                        elbow_high = high_flag
                        main.elbow_low_flag[s]  = low_flag
                        main.elbow_high_flag[s] = high_flag

                    contact_any |= contact
                    flare_soft_active = (main.flare_soft_flag[s]) and arm_vis >= VIS_ARM_GATE     # bool
                    flare_hard_active = (main.flare_hard_flag[s]) and arm_vis >= VIS_ARM_GATE     # bool
                    flare_hard_any   |= flare_hard_active
                    elbow_low_any    |= elbow_low
                    elbow_high_any   |= elbow_high

                    def z_pos(val, mu, sigma, floor, margin=0.10):
                        if val is None or mu is None or sigma is None: return 0.0
                        s_ = max(sigma, floor)
                        z = max(0.0, (val - mu) / s_)
                        return max(0.0, z - margin)

                    # wrist risk score
                    ext_from_neutral = 0.0
                    dev_from_neutral = 0.0
                    if sm_WXT[s] is not None:
                        ext_from_neutral = max(0.0, abs(sm_WXT[s] - (base_WXT[s] or 0.0)) - GRACE_EXT)
                    if sm_WDEV[s] is not None:
                        dev_from_neutral = max(0.0, abs(sm_WDEV[s] - (base_WDEV[s] or 0.0)) - GRACE_DEV)

                    z_wxt  = z_pos(ext_from_neutral,  mu_WXT[s],  sigma_WXT[s],  max(SIGMA_FLOOR_WXT, 4.0))
                    z_wdev = z_pos(dev_from_neutral, mu_WDEV[s], sigma_WDEV[s], max(SIGMA_FLOOR_WDEV, 4.0))
                    WRISK = 0.6 * z_wxt + 0.4 * z_wdev                           # float

                    # quick trip if “hard” for a while
                    if sm_WXT[s] is not None and abs(sm_WXT[s]) >= WXT_HARD: hard_WXT[s] += 1
                    else: hard_WXT[s] = 0
                    if sm_WDEV[s] is not None and abs(sm_WDEV[s]) >= WDEV_HARD: hard_WDEV[s] += 1
                    else: hard_WDEV[s] = 0
                    quick = (hard_WXT[s] >= 12 or hard_WDEV[s] >= 12)            # bool

                    # only judge wrists in typing context
                    context_ok = (typing_active or contact or recent_typing)     # bool

                    # “good lock”: hand is near plane and trunk is GOOD
                    good_lock = (
                        state == "GOOD" and tlen is not None and
                        keyboard_plane_y[s] is not None and sm_hand_px[s] is not None and
                        abs((sm_hand_px[s][1] - keyboard_plane_y[s]) / tlen) <= HAND_NEAR_PLANE_NORM and
                        not quick
                    )

                    # wrist FSM
                    if wr_state[s] == "OK":
                        trigger = (context_ok and WRISK >= WR_ENTER_Z and not good_lock) or (quick and context_ok)
                        if trigger: wr_enter[s] += 1
                        else: wr_enter[s] = 0
                        needed = WR_FRAMES_ENTER + (1 if s == far_arm else 0)
                        if wr_enter[s] >= needed:
                            wr_state[s] = "AT_RISK"; wr_enter[s] = wr_exit[s] = 0
                    else:
                        if WRISK < WR_EXIT_Z or not context_ok or good_lock: wr_exit[s] += 1
                        else: wr_exit[s] = 0
                        if wr_exit[s] >= WR_FRAMES_EXIT:
                            wr_state[s] = "OK"; wr_enter[s] = wr_exit[s] = 0

                    # arm composite state
                    arm_state = "OK"
                    if arm_vis >= VIS_ARM_GATE:
                        if flare_hard_active or (s == 'R' and flare_soft_active): arm_state = "AT_RISK"
                        if fore_hard: arm_state = "AT_RISK"
                        if elbow_low or elbow_high: arm_state = "AT_RISK"
                        if wr_state[s] == "AT_RISK": arm_state = "AT_RISK"; wr_bad_any = True
                        if contact: arm_state = "AT_RISK"
                    all_visible_arms_ok &= (arm_state == "OK")

                    label = "NEAR" if s == near_arm else "FAR"                   # str

                    # draw arm lines + tags
                    up_col = (0, 0, 255) if (flare_hard_active or (s == 'R' and flare_soft_active)) else (160, 210, 255)
                    cv2.line(frame, (int(sh_s[0]), int(sh_s[1])), (int(el_s[0]), int(el_s[1])), up_col, 3, cv2.LINE_AA)
                    if flare_soft_active or fore_soft:
                        tag = "FLARE" if flare_soft_active else "TILT"
                        col = (0, 0, 255) if (flare_hard_active or fore_hard or (s == 'R' and flare_soft_active)) else (0, 180, 240)
                        cv2.putText(frame, tag, (int(el_s[0] + 8), int(el_s[1] - 12)), FONT, 0.55, col, 2, cv2.LINE_AA)
                    if sm_hand_px[s] is not None:
                        col = line_col("OK" if arm_state == "OK" else "FIX")
                        cv2.line(frame, (int(el_s[0]), int(el_s[1])), (int(sm_hand_px[s][0]), int(sm_hand_px[s][1])), col, 4, cv2.LINE_AA)
                        mid = 0.5 * (el_s + sm_hand_px[s])
                        cv2.putText(frame, f"{label} ARM {'GOOD' if arm_state=='OK' else 'FIX'}",
                                    (int(mid[0] + 8), int(mid[1] - 10)), FONT, 0.55, col, 2, cv2.LINE_AA)
                    cv2.circle(frame, (int(el_s[0]), int(el_s[1])), 6, (180, 200, 255), -1, cv2.LINE_AA)
                    if sm_hand_px[s] is not None:
                        cv2.circle(frame, (int(sm_hand_px[s][0]), int(sm_hand_px[s][1])), 7, (255, 220, 120), -1, cv2.LINE_AA)
                    if wr_s is not None:
                        cv2.circle(frame, (int(wr_s[0]), int(wr_s[1])), 4, (180, 180, 180), -1, cv2.LINE_AA)

                    # stash for advanced HUD
                    if not hasattr(main, "armhud"): main.armhud = {}
                    main.armhud[s] = {
                        "ext": sm_WXT[s], "dev": sm_WDEV[s], "flare": flare_angle,
                        "fore_deg": fore_deg,
                        "stdY": typing_stats[s]["stdY"], "typing": typing_stats[s]["typing"],
                        "contact": contact, "recent": recent_typing,
                        "elbow_low": elbow_low, "elbow_high": elbow_high,
                        "vis": arm_vis,
                    }

            # ---------- HUD / header ----------
            if pose_present:
                draw_panel(frame, (24, 22), (1240, 140))
                trunk_good = (state == "GOOD")                                           # bool
                any_arm_visible = 'any_arm_visible' in locals() and any_arm_visible      # bool
                all_visible_arms_ok = 'all_visible_arms_ok' in locals() and all_visible_arms_ok  # bool
                overall_good = trunk_good and (all_visible_arms_ok if any_arm_visible else True) # bool

                if session_started:
                    frames_total += 1
                    if overall_good: frames_good += 1
                    pct = 100.0 * frames_good / max(1, frames_total)                     # float: % good
                    elapsed = int(time.time() - session_start_time); mm, ss = divmod(elapsed, 60)  # int,int
                else:
                    pct = 0.0; mm = ss = 0

                suggestion = simple_suggestion(
                    trunk_bad=(not trunk_good),
                    flare_hard_any=('flare_hard_any' in locals() and flare_hard_any),
                    elbow_low_any=('elbow_low_any' in locals() and elbow_low_any),
                    elbow_high_any=('elbow_high_any' in locals() and elbow_high_any),
                    forearm_bad_any=forearm_bad_any,
                    wr_bad_any=('wr_bad_any' in locals() and wr_bad_any),
                    contact_any=('contact_any' in locals() and contact_any)
                )

                status_txt = "GOOD" if overall_good else "ADJUST"                        # str
                status_col = COLOR_GOOD if overall_good else COLOR_BAD                   # BGR
                cv2.putText(frame, f"Status: {status_txt}", (46, 66), FONT, 1.1, status_col, 4, cv2.LINE_AA)
                cv2.putText(frame, f"Tip: {suggestion}", (46, 110), FONT, 0.80,
                            COLOR_WARN if not overall_good else COLOR_INFO, 2, cv2.LINE_AA)

                right_x = 860                                                            # int: HUD column x
                cv2.putText(frame, f"Session: {mm:02d}:{ss:02d}", (right_x, 66), FONT, 0.9, (230, 230, 230), 2, cv2.LINE_AA)
                cv2.putText(frame, f"% Good: {pct:5.1f}%", (right_x, 110), FONT, 0.9, (230, 230, 230), 2, cv2.LINE_AA)
            else:
                # Clean header when no pose
                draw_panel(frame, (24, 22), (1240, 140))
                cv2.putText(frame, "No pose detected", (46, 66), FONT, 1.1, COLOR_BAD, 4, cv2.LINE_AA)
                cv2.putText(frame, "Tip: Check camera connection, angle & lighting",
                            (46, 110), FONT, 0.80, COLOR_INFO, 2, cv2.LINE_AA)

            # Advanced panel — compact; show NEAR on left, FAR on right
            if advanced_mode and hasattr(main, "armhud"):
                if 'near_arm' in locals():
                    nf_order = [near_arm, ('R' if near_arm == 'L' else 'L')]             # list[str]
                else:
                    nf_order = ['L', 'R']
                labels = {nf_order[0]: "Near", nf_order[1]: "Far"}                       # dict[str,str]

                y0 = 160; draw_panel(frame, (24, y0), (1240, y0 + 220))                  # int: top y
                col1_x = 46; col2_x = 640; line_h = 24; i1 = 0; i2 = 0; deg = "°"        # ints, str

                def putL(t):
                    nonlocal i1
                    cv2.putText(frame, t, (col1_x, y0 + 36 + i1*line_h), FONT, 0.58, (230, 230, 230), 2, cv2.LINE_AA)
                    i1 += 1

                def putR(t):
                    nonlocal i2
                    cv2.putText(frame, t, (col2_x, y0 + 36 + i2*line_h), FONT, 0.58, (230, 230, 230), 2, cv2.LINE_AA)
                    i2 += 1

                putL("[ADVANCED]")
                putL(f"TFA {safe_fmt(sm_TFA if 'sm_TFA' in locals() else None, deg)} | "
                     f"NSS {safe_fmt(sm_NSS if 'sm_NSS' in locals() else None, deg)}")
                putL(f"RFH {safe_fmt(sm_RFH if 'sm_RFH' in locals() else None, 'torso')} | "
                     f"HDT {safe_fmt(sm_HDT if 'sm_HDT' in locals() else None, deg)}")
                if 'trunk_len' in locals() and trunk_len and base_TLEN:
                    dr = max(0.0, 1.0 - trunk_len/max(1e-6, base_TLEN)) * 100.0          # float %
                    putL(f"Shoulder drop: {dr:.1f}% (vs. baseline)")
                pd = float(np.median(np.array([y for y in keyboard_plane_y.values() if y is not None], np.float32))) \
                     if any(y is not None for y in keyboard_plane_y.values()) else None  # float|None
                putL(f"Plane y {pd:.1f}px" if pd is not None else "Plane y N/A")
                for s, put in [(nf_order[0], putL), (nf_order[1], putR)]:
                    a = getattr(main, "armhud", {}).get(s, {})                            # dict
                    t = "T" if a.get("typing") else "t"
                    r = "R" if a.get("recent") else "r"
                    c = "C" if a.get("contact") else "c"
                    el = "low" if a.get("elbow_low") else "-"
                    eh = "high" if a.get("elbow_high") else "-"
                    ext = safe_fmt(a.get("ext"), deg); dev = safe_fmt(a.get("dev"), deg); fl = safe_fmt(a.get("flare"), deg)
                    fore = safe_fmt(a.get("fore_deg"), deg); vis = f"{a.get('vis', 0):.2f}"
                    put(f"{labels[s]}: ext {ext} | dev {dev} | flare {fl} | fore {fore}")
                    put(f"    stdY {safe_fmt(a.get('stdY'), 'px')} | {t}{r}{c} | {el}/{eh} | vis {vis}")

            # Footer & FPS
            cv2.putText(frame, "r: recalibrate   a: advanced   q: quit",
                        (36, h-24), FONT, 0.65, (230, 230, 230), 2, cv2.LINE_AA)
            if SHOW_FPS:
                cv2.putText(frame, f"FPS: {fps:.1f}", (w-180, 40), FONT, 0.62, (210, 210, 210), 2, cv2.LINE_AA)

            cv2.imshow("Posture Monitoring", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # full reset (post-cal)
                calibrating = True; calibration_start = time.time(); calibration_frames = 0
                calib_TFA.clear(); calib_RFH.clear(); calib_HDT.clear(); calib_NSS.clear(); calib_TLEN.clear()
                base_TFA = base_RFH = base_HDT = base_NSS = None; base_TLEN = None
                mu_TFA = sigma_TFA = mu_RFH = sigma_RFH = None
                mu_HDT = sigma_HDT = mu_NSS = sigma_NSS = None
                sm_TFA = sm_RFH = sm_HDT = sm_NSS = None
                hist_TFA.clear(); hist_RFH.clear(); hist_HDT.clear(); hist_NSS.clear()
                state = "GOOD"; enter_count = exit_count = 0

                neck_base_px = None; hip_base_px = None; neck_miss = hip_miss = 0

                for s in sides:
                    calib_WXT[s].clear(); calib_WDEV[s].clear(); calib_wristY[s].clear()
                    mu_WXT[s] = mu_WDEV[s] = None; sigma_WXT[s] = sigma_WDEV[s] = None
                    base_WXT[s] = base_WDEV[s] = None
                    sm_WXT[s] = sm_WDEV[s] = None
                    hist_WXT[s].clear(); hist_WDEV[s].clear()
                    wr_state[s] = "OK"; wr_enter[s] = wr_exit[s] = 0
                    hard_WXT[s] = hard_WDEV[s] = 0
                    wrY_hist[s].clear()
                    typing_stats[s]["stdY"] = 0.0; typing_stats[s]["typing"] = False; last_typing_time[s] = 0.0
                    sm_hand_px[s] = None; hand_miss[s] = 0
                    last_good_elbow[s] = None; elbow_miss_count[s] = 0
                    keyboard_plane_y[s] = None

                session_started = False; session_start_time = None; frames_total = frames_good = 0

            elif key in (ord('a'), ord('A')):
                advanced_mode = not advanced_mode

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
