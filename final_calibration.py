import numpy as np
import cv2
import pyrealsense2 as rs
import json
from rtde_receive import RTDEReceiveInterface

# -----------------------------
# CONFIG
# -----------------------------
ROBOT_IP    = "192.168.56.101"
NUM_POINTS  = 10
OUTPUT_FILE = "calibration.json"
MIN_DEPTH   = 0.1
MAX_DEPTH   = 2.0
DEPTH_PATCH = 5

# -----------------------------
# ROBOT
# -----------------------------
rtde_r = RTDEReceiveInterface(ROBOT_IP)

# -----------------------------
# REALSENSE SETUP
# -----------------------------
print("[INFO] Starting RealSense...")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
align = rs.align(rs.stream.color)

color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intrinsics = color_stream.get_intrinsics()

camera_matrix = np.array([
    [intrinsics.fx, 0,             intrinsics.ppx],
    [0,             intrinsics.fy, intrinsics.ppy],
    [0,             0,             1             ]
])

dist_coeffs = np.array(intrinsics.coeffs).reshape(5, 1)

print(f"[INFO] Camera intrinsics loaded — fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f}")
print(f"[INFO] Distortion: {intrinsics.coeffs}")
print("[INFO] Camera ready")

# -----------------------------
# STORAGE
# -----------------------------
camera_points = []
robot_points  = []

current_depth_frame = None

# -----------------------------
# DEPTH HELPER
# Median depth over a patch to reduce noise from single pixel readings
# -----------------------------
def pixel_to_3d(x, y, depth_frame):
    half = DEPTH_PATCH // 2
    depths = []

    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            px, py = x + dx, y + dy
            if 0 <= px < 640 and 0 <= py < 480:
                d = depth_frame.get_distance(px, py)
                if MIN_DEPTH < d < MAX_DEPTH:
                    depths.append(d)

    if not depths:
        print("[WARN] No valid depth at click location")
        return None

    depth = float(np.median(depths))
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
    return np.array(point)

# -----------------------------
# MOUSE CALLBACK
# Left click on the laser dot to record a calibration sample
# -----------------------------
def mouse_callback(event, x, y, flags, param):
    global camera_points, robot_points, current_depth_frame

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if current_depth_frame is None:
        print("[WARN] No depth frame available yet")
        return

    if len(camera_points) >= NUM_POINTS:
        print("[INFO] Already collected enough points")
        return

    P_camera = pixel_to_3d(x, y, current_depth_frame)
    if P_camera is None:
        return

    tcp = rtde_r.getActualTCPPose()
    P_robot = np.array(tcp[:3])

    camera_points.append(P_camera)
    robot_points.append(P_robot)

    n = len(camera_points)
    print(f"[SAVED] Point {n}/{NUM_POINTS}")
    print(f"  Camera : {P_camera}")
    print(f"  Robot  : {P_robot}")

# -----------------------------
# SVD CALIBRATION
# Solves for R and t such that P_robot = R @ P_camera + t
# -----------------------------
def calibrate(camera_points, robot_points):
    A = np.array(camera_points)
    B = np.array(robot_points)

    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # fix reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    return R, t

# -----------------------------
# SAVE
# -----------------------------
def save_calibration(R, t):
    data = {
        "R": R.tolist(),
        "t": t.tolist(),
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=4)
    print(f"\n[SAVED] Calibration written to {OUTPUT_FILE}")

# -----------------------------
# WINDOW + CALLBACK
# -----------------------------
cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Calibration", mouse_callback)

print("\n[INFO] Instructions:")
print("  - Move the robot so the laser dot lands on a known surface point")
print("  - LEFT CLICK on the laser dot in the image to record that sample")
print("  - Collect samples across the full workspace (vary x, y, and height)")
print("  - Press ESC to exit early\n")

# -----------------------------
# MAIN LOOP
# -----------------------------
try:
    while len(camera_points) < NUM_POINTS:

        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        current_depth_frame = depth_frame

        image = np.asanyarray(color_frame.get_data())

        # draw previously recorded points on screen
        for i, (cam_pt,) in enumerate(zip(camera_points)):
            px = int(intrinsics.ppx + intrinsics.fx * cam_pt[0] / cam_pt[2])
            py = int(intrinsics.ppy + intrinsics.fy * cam_pt[1] / cam_pt[2])
            cv2.circle(image, (px, py), 6, (0, 255, 0), -1)
            cv2.putText(image, str(i + 1), (px + 8, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(image,
                    f"Points: {len(camera_points)}/{NUM_POINTS}  |  Click laser dot to record",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Calibration", image)

        if cv2.waitKey(1) == 27:
            print("[EXIT] ESC pressed")
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# -----------------------------
# CALIBRATION STEP
# -----------------------------
if len(camera_points) >= 4:
    print(f"\n[INFO] Computing calibration from {len(camera_points)} points...")

    R, t = calibrate(camera_points, robot_points)
    save_calibration(R, t)

    print("\n[DONE] Calibration complete")
    print(f"  R =\n{R}")
    print(f"  t = {t}")

else:
    print(f"[WARN] Only {len(camera_points)} points collected — need at least 4. Calibration skipped.")