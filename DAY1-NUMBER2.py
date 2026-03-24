import cv2
import numpy as np

# ================== CONFIG ==================

VIDEO_PATH = "video1.avi"
LOG_PATH = "user1.txt"

# --- ROI ---
ROI_TOP_RATIO = 0.60

# --- Предобработка ---
USE_CONTRAST = False
ALPHA_CONTRAST = 1.2
BETA_BRIGHTNESS = 0

BLUR_KERNEL = 5
USE_OTSU = False
THRESHOLD = 170

# --- Морфология ---
MORPH_KERNEL_SIZE = 5
USE_MORPH_OPEN = True
USE_MORPH_CLOSE = True

# --- Моторы ---
BASE_SPEED = 170
MIN_SPEED = 0
MAX_SPEED = 255

KP = 35.0
KD = 8.0
STEER_MULTIPLIER = 1.0
MAX_STEER = 50

# --- Направление ---
INVERT_ERROR = False
SWAP_MOTORS = True

# --- Линия ---
USE_LAST_ERROR = False
STOP_IF_LINE_NOT_FOUND = True
MIN_LINE_AREA = 300

# --- Проверка движения ---
USE_MOTION_CHECK = True
MOTION_THRESHOLD = 0.55        # порог среднего изменения яркости
MOTION_BLUR_KERNEL = 5          # blur перед сравнением кадров
MOTION_USE_ROI_ONLY = True      # сравнивать только ROI
SKIP_MOTION_ON_FIRST_FRAME = True

# --- Доп. логика нулей ---
USE_ZERO_DEADBAND = False
ZERO_ERROR_DEADBAND = 0.03
ZERO_AREA_FOR_STOP = 450

# --- Отладка ---
SHOW_WINDOWS = True
PRINT_DEBUG = False

# ===========================================


def apply_contrast(frame: np.ndarray) -> np.ndarray:
    if not USE_CONTRAST:
        return frame
    return cv2.convertScaleAbs(frame, alpha=ALPHA_CONTRAST, beta=BETA_BRIGHTNESS)


def preprocess(roi: np.ndarray) -> np.ndarray:
    roi = apply_contrast(roi)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    if BLUR_KERNEL and BLUR_KERNEL > 0:
        k = BLUR_KERNEL
        if k % 2 == 0:
            k += 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    if USE_OTSU:
        _, mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        _, mask = cv2.threshold(
            gray, THRESHOLD, 255, cv2.THRESH_BINARY_INV
        )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )

    if USE_MORPH_OPEN:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if USE_MORPH_CLOSE:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def find_main_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, None, 0.0, None, None

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))

    if area < MIN_LINE_AREA:
        return False, contour, area, None, None

    M = cv2.moments(contour)
    if M["m00"] == 0:
        return False, contour, area, None, None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return True, contour, area, cx, cy


def compute_error(cx: int, roi_w: int) -> float:
    center_x = roi_w // 2
    error = (cx - center_x) / (roi_w / 2)

    if INVERT_ERROR:
        error = -error

    return error


LEFT_OFFSET = -1 # <-- вот это добавили


def compute_motor_speeds(error: float, prev_error: float):
    d_error = error - prev_error

    steer = KP * error + KD * d_error
    steer *= STEER_MULTIPLIER
    steer = max(-MAX_STEER, min(MAX_STEER, steer))

    left_raw = int(round(BASE_SPEED - steer))
    right_raw = int(round(BASE_SPEED + steer))

    if SWAP_MOTORS:
        left_raw, right_raw = right_raw, left_raw

    # 🔥 ВАЖНО: добавляем после swap
    left_raw += LEFT_OFFSET
    right_raw -= LEFT_OFFSET

    left = max(MIN_SPEED, min(MAX_SPEED, left_raw))
    right = max(MIN_SPEED, min(MAX_SPEED, right_raw))

    return left, right, steer


def prepare_motion_frame(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if MOTION_BLUR_KERNEL and MOTION_BLUR_KERNEL > 0:
        k = MOTION_BLUR_KERNEL
        if k % 2 == 0:
            k += 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    return gray


def detect_motion(prev_frame_for_motion: np.ndarray, curr_frame_for_motion: np.ndarray):
    if prev_frame_for_motion is None:
        return True, 999.0

    prev_gray = prepare_motion_frame(prev_frame_for_motion)
    curr_gray = prepare_motion_frame(curr_frame_for_motion)

    diff = cv2.absdiff(curr_gray, prev_gray)
    motion_score = float(np.mean(diff))
    is_moving = motion_score >= MOTION_THRESHOLD

    return is_moving, motion_score


def should_output_zero(line_found: bool, line_area: float, error: float, is_moving: bool) -> bool:
    if USE_MOTION_CHECK and not is_moving:
        return True

    if STOP_IF_LINE_NOT_FOUND and not line_found:
        return True

    if USE_ZERO_DEADBAND and line_found:
        if abs(error) < ZERO_ERROR_DEADBAND and line_area < ZERO_AREA_FOR_STOP:
            return True

    return False


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Не удалось открыть видео: {VIDEO_PATH}")
        return

    results = []
    frame_id = 0
    prev_error = 0.0
    last_error = 0.0
    prev_motion_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        h, w = frame.shape[:2]
        y1 = int(h * ROI_TOP_RATIO)
        roi = frame[y1:, :]
        roi_h, roi_w = roi.shape[:2]

        mask = preprocess(roi)
        line_found, contour, line_area, cx, cy = find_main_contour(mask)

        if line_found and cx is not None:
            error = compute_error(cx, roi_w)
            last_error = error
        else:
            error = last_error if USE_LAST_ERROR else 0.0
            cx = roi_w // 2
            cy = roi_h // 2

        if USE_MOTION_CHECK:
            current_motion_frame = roi if MOTION_USE_ROI_ONLY else frame

            if prev_motion_frame is None and SKIP_MOTION_ON_FIRST_FRAME:
                is_moving = True
                motion_score = 999.0
            else:
                is_moving, motion_score = detect_motion(prev_motion_frame, current_motion_frame)

            prev_motion_frame = current_motion_frame.copy()
        else:
            is_moving = True
            motion_score = 999.0

        output_zero = should_output_zero(line_found, line_area, error, is_moving)

        if output_zero:
            left_speed = 0
            right_speed = 0
            steer = 0.0
        else:
            left_speed, right_speed, steer = compute_motor_speeds(error, prev_error)

        if PRINT_DEBUG:
            print(
                f"frame={frame_id:03d} "
                f"moving={is_moving} "
                f"motion={motion_score:.2f} "
                f"found={line_found} "
                f"area={line_area:.1f} "
                f"error={error:.3f} "
                f"steer={steer:.2f} "
                f"L={left_speed} R={right_speed}"
            )

        results.append((frame_id, left_speed, right_speed))
        prev_error = error

        if SHOW_WINDOWS:
            vis = roi.copy()

            cv2.line(vis, (roi_w // 2, 0), (roi_w // 2, roi_h), (255, 0, 0), 2)

            if contour is not None:
                cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)

            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)

            cv2.putText(
                vis, f"frame={frame_id}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            cv2.putText(
                vis, f"area={int(line_area)}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            cv2.putText(
                vis, f"err={error:.3f}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            cv2.putText(
                vis, f"motion={motion_score:.2f}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
            cv2.putText(
                vis, f"moving={is_moving}", (10, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
            cv2.putText(
                vis, f"L={left_speed} R={right_speed}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

            cv2.imshow("mask", mask)
            cv2.imshow("vis", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        for frame_id, left_speed, right_speed in results:
            f.write(f"{frame_id} {left_speed} {right_speed}\n")

    print(f"Готово. Лог сохранен в: {LOG_PATH}")


if __name__ == "__main__":
    main()