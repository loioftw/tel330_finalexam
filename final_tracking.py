import cv2
import numpy as np
import pyrealsense2 as rs
import json
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from filterpy.kalman import KalmanFilter

# -----------------------------
# CONFIG
# -----------------------------
ROBOT_IP         = "192.168.56.101"
CALIBRATION_FILE = "calibration.json"

HOME_POSITION     = [-0.4661, 0.9194, 0.2494, 3.14, 0, 0.0]
FIXED_ORIENTATION = [3.14, 0.0, 0.0]

WORKSPACE = {
    "x": (-1,  1),
    "y": (-1,  1),
}

# servoL parameters
SERVO_VEL       = 0.2
SERVO_ACC       = 0.2
SERVO_TIMESTEP  = 0.008
SERVO_LOOKAHEAD = 0.05
SERVO_GAIN      = 150

# Detection
MIN_CONTOUR_AREA = 150
DEPTH_PATCH      = 7
MIN_DEPTH        = 0.05
MAX_DEPTH        = 2.0

# Kalmam filter
# higher = reacts faster but noisier, lower = smoother but slower
PROCESS_NOISE     = 0.08

# higher = trust measurements less, lower = trust measurements more
MEASUREMENT_NOISE = 0.005

PREDICT_HORIZON   = SERVO_LOOKAHEAD

# -----------------------------
# LOAD CALIBRATION
# -----------------------------
try:
    with open(CALIBRATION_FILE) as f:
        cal = json.load(f)
    R_cam2rob = np.array(cal["R"])
    t_cam2rob = np.array(cal["t"])
    print(f"[INFO] Calibration loaded from {CALIBRATION_FILE}")
    print(f"  Mean error was: {cal.get('mean_error_mm', 'n/a')} mm")
except FileNotFoundError:
    raise RuntimeError(
        f"Calibration file '{CALIBRATION_FILE}' not found. "
        "Run calibration.py first."
    )

# -----------------------------
# ROBOT
# -----------------------------
print("[INFO] Connecting to robot...")
rtde_c = RTDEControlInterface(ROBOT_IP)
rtde_r = RTDEReceiveInterface(ROBOT_IP)
print("[INFO] Robot connected")

# -----------------------------
# REALSENSE
# -----------------------------
print("[INFO] Starting RealSense...")
pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  60)

profile    = pipeline.start(config)
align      = rs.align(rs.stream.color)
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
print("[INFO] Camera ready")

# -----------------------------
# DEPTH HELPER
# Takes the median depth over a patch of pixels to reduce noise
# -----------------------------
def pixel_to_3d(x, y, depth_frame):
    half   = DEPTH_PATCH // 2
    depths = []
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            px, py = x + dx, y + dy
            if 0 <= px < 640 and 0 <= py < 480:
                d = depth_frame.get_distance(px, py)
                if MIN_DEPTH < d < MAX_DEPTH:
                    depths.append(d)
    if not depths:
        return None
    depth = float(np.median(depths))
    return np.array(rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth))

# -----------------------------
# COORDINATE TRANSFORMS
# -----------------------------
def camera_to_robot(P_camera):
    return R_cam2rob @ P_camera + t_cam2rob

LASER_HEIGHT = 0.3

def build_pose(P_robot):
    return [
        P_robot[0], P_robot[1], LASER_HEIGHT,
        FIXED_ORIENTATION[0],
        FIXED_ORIENTATION[1],
        FIXED_ORIENTATION[2],
    ]

# -----------------------------
# WORKSPACE SAFETY CHECK
# -----------------------------
def within_workspace(P_robot):
    for val, (lo, hi) in zip(P_robot, [WORKSPACE["x"], WORKSPACE["y"]]):
        if not (lo <= val <= hi):
            return False
    return True

# -----------------------------
# OBJECT DETECTION
# Detects red/orange objects using HSV thresholding
# Red wraps around in HSV so two masks are needed and combined
# -----------------------------
def detect_object(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, np.array([0,   80, 150]), np.array([15,  255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 80, 150]), np.array([180, 255, 255]))
    mask  = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, mask

    valid = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    if not valid:
        return None, None, mask

    fish = max(valid, key=cv2.contourArea)
    M    = cv2.moments(fish)
    if M["m00"] == 0:
        return None, None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return fish, (cx, cy), mask

# -----------------------------
# KALMAN FILTER
# Constant-velocity model tracking x, y position and velocity
# State vector:  [x, y, vx, vy]
# Measurement:   [x, y]
# -----------------------------
def make_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)

    dt = SERVO_TIMESTEP

    kf.F = np.array([
        [1, 0, dt, 0],
        [0, 1,  0, dt],
        [0, 0,  1, 0],
        [0, 0,  0, 1],
    ])

    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ])

    kf.Q = np.eye(4) * PROCESS_NOISE
    kf.R = np.eye(2) * MEASUREMENT_NOISE
    kf.P = np.eye(4) * 1.0
    kf.x = np.zeros((4, 1))

    return kf


kf             = make_kalman_filter()
kf_initialised = False


def kf_reset():
    global kf, kf_initialised
    kf             = make_kalman_filter()
    kf_initialised = False


def kf_update(x_rob, y_rob, dt_actual):
    global kf_initialised

    kf.F[0, 2] = dt_actual
    kf.F[1, 3] = dt_actual

    z = np.array([[x_rob], [y_rob]])

    if not kf_initialised:
        kf.x = np.array([[x_rob], [y_rob], [0.0], [0.0]])
        kf_initialised = True
    else:
        kf.predict()

    kf.update(z)
    return kf.x.flatten()


def kf_predict_ahead(horizon_s):
    x, y, vx, vy = kf.x.flatten()
    return np.array([x + vx * horizon_s,
                     y + vy * horizon_s])


# -----------------------------
# MAIN LOOP
# -----------------------------
cv2.namedWindow("Fish Tracking", cv2.WINDOW_NORMAL)

lost_frames = 0
MAX_LOST    = 10
prev_time   = None

# -----------------------------
# STARTUP DELAY
# Wait for camera to stabilise before sending any robot commands
# -----------------------------
STARTUP_DELAY = 3.0

print(f"[INFO] Warming up for {STARTUP_DELAY}s — robot will not move yet")
start_time = cv2.getTickCount()

while True:
    elapsed   = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    remaining = STARTUP_DELAY - elapsed

    frames  = pipeline.wait_for_frames()
    aligned = align.process(frames)
    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())
    vis   = frame.copy()

    fish, center, mask = detect_object(frame)
    if fish is not None and center is not None:
        cv2.drawContours(vis, [fish], -1, (0, 255, 0), 2)
        cv2.circle(vis, center, 5, (255, 0, 0), -1)

    cv2.putText(vis,
                f"STARTING IN {remaining:.1f}s — check detection",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    cv2.imshow("Fish Tracking", vis)

    if remaining <= 0:
        break
    if cv2.waitKey(1) == 27:
        pipeline.stop()
        cv2.destroyAllWindows()
        exit()

print("[INFO] Tracking started — press ESC to exit")

try:
    while True:

        frames  = pipeline.wait_for_frames()
        aligned = align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # actual dt between frames keeps velocity estimates accurate
        now = cv2.getTickCount() / cv2.getTickFrequency()
        dt  = (now - prev_time) if prev_time is not None else SERVO_TIMESTEP
        dt  = np.clip(dt, 0.001, 0.1)
        prev_time = now

        frame = np.asanyarray(color_frame.get_data())
        vis   = frame.copy()

        fish, center, mask = detect_object(frame)

        # -------------------------
        # OBJECT LOST
        # -------------------------
        if fish is None or center is None:
            lost_frames += 1
            cv2.putText(vis, "LOST", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            if lost_frames >= MAX_LOST:
                rtde_c.servoStop()
                kf_reset()
                if lost_frames == MAX_LOST:
                    print("[INFO] Object lost — moving to home position")
                    rtde_c.moveL(HOME_POSITION, 1.0, 1.0)

            cv2.imshow("Fish Tracking", vis)
            if cv2.waitKey(1) == 27:
                break
            continue

        lost_frames = 0
        cx, cy      = center

        # pixel to 3D point in camera frame
        P_camera = pixel_to_3d(cx, cy, depth_frame)
        if P_camera is None:
            cv2.imshow("Fish Tracking", vis)
            if cv2.waitKey(1) == 27:
                break
            continue

        # transform to robot base frame
        P_robot_raw = camera_to_robot(P_camera)

        if not within_workspace(P_robot_raw):
            cv2.putText(vis, "OUT OF WORKSPACE", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Fish Tracking", vis)
            if cv2.waitKey(1) == 27:
                break
            continue

        # update Kalman filter and predict where the fish will be
        state    = kf_update(P_robot_raw[0], P_robot_raw[1], dt)
        P_filt   = state[:2]
        P_pred   = kf_predict_ahead(PREDICT_HORIZON)
        velocity = state[2:]

        # send predicted position to robot
        pose = build_pose(P_pred)
        rtde_c.servoL(pose, SERVO_VEL, SERVO_ACC, SERVO_TIMESTEP,
                      SERVO_LOOKAHEAD, SERVO_GAIN)

        # -------------------------
        # VISUALISATION
        # -------------------------
        cv2.drawContours(vis, [fish], -1, (0, 255, 0), 2)
        cv2.circle(vis, (cx, cy), 5, (255, 0, 0), -1)

        cv2.putText(vis,
                    f"Cam:   {P_camera[0]:.3f} {P_camera[1]:.3f} {P_camera[2]:.3f} m",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(vis,
                    f"Filt:  {P_filt[0]:.3f} {P_filt[1]:.3f} m",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.putText(vis,
                    f"Pred:  {P_pred[0]:.3f} {P_pred[1]:.3f} m  (t+{PREDICT_HORIZON*1000:.0f}ms)",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1)
        cv2.putText(vis,
                    f"Vel:   {velocity[0]*1000:.1f} {velocity[1]*1000:.1f} mm/s",
                    (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1)
        cv2.putText(vis,
                    f"Area:  {cv2.contourArea(fish):.0f} px2",
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

        cv2.imshow("Fish Tracking", vis)

        if cv2.waitKey(1) == 27:
            print("[EXIT] ESC pressed")
            break

finally:
    print("[INFO] Stopping robot and camera...")
    rtde_c.servoStop()
    rtde_c.stopScript()
    pipeline.stop()
    cv2.destroyAllWindows()
    print("[INFO] Done")