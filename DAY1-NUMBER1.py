import cv2
import numpy as np
from collections import deque
import os

VIDEO_FOLDER = r"C:\Users\kvantorium33\Desktop\хактон\Day1\TraficControl\for_pub"
VIDEOS = [
    "robot_2026-03-13_11-11-07.avi",
    "robot_2026-03-13_11-13-26.avi",
    "robot_2026-03-13_11-16-27.avi"
]

class SimpleDetector:
    def __init__(self):
        self.traffic_history = deque(maxlen=8)
        self.line_history = deque(maxlen=5)
        self.stable_traffic = "unknown"
        self.prev_traffic = "unknown"
        self.green_switch_frames = 0  # счётчик кадров после переключения на зелёный
        self.line_was_seen = False  # была ли линия видна недавно

    def get_traffic_color(self, frame):
        h, w = frame.shape[:2]
        x1 = int(w * 0.60)
        x2 = int(w * 0.98)
        y1 = int(h * 0.05)
        y2 = int(h * 0.55)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Красный
        red1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        red2 = cv2.inRange(hsv, (160, 50, 50), (179, 255, 255))
        red_mask = cv2.bitwise_or(red1, red2)
        red_score = cv2.countNonZero(red_mask)

        # Зелёный
        green_mask = cv2.inRange(hsv, (35, 50, 50), (90, 255, 255))
        green_score = cv2.countNonZero(green_mask)

        # Определяем цвет
        if red_score > 150 and red_score > green_score:
            current = "red"
        elif green_score > 150 and green_score > red_score:
            current = "green"
        else:
            current = "unknown"

        # Стабилизация
        self.traffic_history.append(current)
        if self.traffic_history.count("red") >= 5:
            self.stable_traffic = "red"
        elif self.traffic_history.count("green") >= 5:
            self.stable_traffic = "green"

        # Отслеживаем переключение на зелёный
        if self.prev_traffic != "green" and self.stable_traffic == "green":
            self.green_switch_frames = 30  # будет показывать GO! следующие 30 кадров
        self.prev_traffic = self.stable_traffic

        if self.green_switch_frames > 0:
            self.green_switch_frames -= 1

        # Отрисовка
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f"R:{red_score} G:{green_score}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return self.stable_traffic

    def get_stop_line(self, frame):
        h, w = frame.shape[:2]
        y_start = int(h * 0.70)
        bottom = frame[y_start:h, :]
        if bottom.size == 0:
            return False

        hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, (15, 70, 70), (35, 255, 255))

        kernel = np.ones((5, 5), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                if w_box > h_box * 2:
                    current = True
                    cv2.rectangle(bottom, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

        self.line_history.append(current)
        line_detected = sum(self.line_history) >= 3

        # Запоминаем, была ли линия видна недавно (для GO!)
        if line_detected:
            self.line_was_seen = True
        elif self.line_was_seen and not line_detected:
            # Если линия пропала, держим флаг ещё 20 кадров
            if hasattr(self, 'line_seen_counter'):
                self.line_seen_counter -= 1
                if self.line_seen_counter <= 0:
                    self.line_was_seen = False
            else:
                self.line_seen_counter = 20
        else:
            self.line_was_seen = False

        return line_detected

    def should_show_go(self, line_detected):
        """Показываем GO! если линия видна ИЛИ недавно переключился на зелёный И линия была недавно видна"""
        if line_detected:
            return True
        # Если зелёный переключился недавно и линия была видна в последнее время
        if self.green_switch_frames > 0 and self.line_was_seen:
            return True
        return False

def test_video(video_path, video_name, video_num, total_videos):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Не удалось открыть {video_name}")
        return False

    det = SimpleDetector()
    print(f"\n{'=' * 55}\n🎬 [{video_num}/{total_videos}] {video_name}\n{'=' * 55}")

    frame_count = 0
    red_frames = 0
    green_frames = 0
    line_frames = 0
    go_shown = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        h, w = frame.shape[:2]

        traffic = det.get_traffic_color(frame)
        line = det.get_stop_line(frame)
        show_go = det.should_show_go(line)

        if traffic == "red":
            red_frames += 1
        elif traffic == "green":
            green_frames += 1
        if line:
            line_frames += 1

        # Интерфейс
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        if traffic == "red":
            cv2.putText(frame, "🔴 RED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        elif traffic == "green":
            cv2.putText(frame, "🟢 GREEN", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "⚪ UNKNOWN", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128, 128, 128), 2)

        line_text = "🟡 LINE" if line else "⚫ NO LINE"
        line_color = (0, 255, 0) if line else (100, 100, 100)
        cv2.putText(frame, line_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)

        # Отладка: показываем счётчик переключения
        if det.green_switch_frames > 0:
            cv2.putText(frame, f"GREEN SWITCH: {det.green_switch_frames}", (w - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # STOP / GO!
        if traffic == "red" and line:
            cv2.putText(frame, "STOP", (w // 2 - 70, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5)
        elif traffic == "green" and show_go:
            cv2.putText(frame, "GO!", (w // 2 - 60, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 5)
            go_shown = True

        cv2.imshow(f"TEST: {video_name}", frame)
        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyWindow(f"TEST: {video_name}")

    print(f"📊 Кадров: {frame_count} | 🔴 {red_frames} | 🟢 {green_frames} | 🟡 {line_frames}")
    print(f"   GO! показан: {'✅' if go_shown else '❌'}")
    if red_frames > 0 and green_frames > 0 and line_frames > 0:
        print("   ✅ ОТЛИЧНО!")
        return True
    elif go_shown:
        print("   ✅ GO! показан (даже после исчезновения линии)")
        return True
    elif (red_frames > 0 or green_frames > 0) and line_frames > 0:
        print("   ✅ ХОРОШО!")
        return True
    else:
        print("   ❌ ПРОБЛЕМА")
        return False

print("=" * 55)
print("🚦 ФИНАЛЬНАЯ ВЕРСИЯ С ПОДДЕРЖКОЙ ПОЗДНЕГО ЗЕЛЁНОГО")
print("=" * 55)

results = {}
for i, video in enumerate(VIDEOS, 1):
    video_path = os.path.join(VIDEO_FOLDER, video)
    results[video] = test_video(video_path, video, i, len(VIDEOS))
    if i < len(VIDEOS):
        print("\n➡️ Нажми любую клавишу...")
        cv2.waitKey(0)

cv2.destroyAllWindows()
print("\n🏆 ИТОГ")
for v, s in results.items():
    print(f"   {v}: {'✅' if s else '❌'}")

if all(results.values()):
    print("\n🎉 ГОТОВО К СДАЧЕ!")