# Real-time object detection with YOLOv8 + SQLite logging
#
# Setup:   pip install ultralytics
# Run:     python main.py
# Report:  python report.py
# Press 'q' in the video window to quit.

import sqlite3
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

DB_PATH = "detections.db"
CONF_THRESHOLD = 0.5   # ignore weak detections
LOG_INTERVAL = 1.0     # seconds between database writes (avoids ~30 rows/sec per object)


def init_db():
    """Create the database and tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            ended_at   TEXT
        );

        CREATE TABLE IF NOT EXISTS detections (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  INTEGER NOT NULL,
            detected_at TEXT NOT NULL,
            class_name  TEXT NOT NULL,
            confidence  REAL NOT NULL,
            x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        CREATE INDEX IF NOT EXISTS idx_detections_class
            ON detections(class_name);
        """
    )
    conn.commit()
    return conn


def start_session(conn):
    cur = conn.execute(
        "INSERT INTO sessions (started_at) VALUES (?)",
        (datetime.now().isoformat(timespec="seconds"),),
    )
    conn.commit()
    return cur.lastrowid


def end_session(conn, session_id):
    conn.execute(
        "UPDATE sessions SET ended_at = ? WHERE id = ?",
        (datetime.now().isoformat(timespec="seconds"), session_id),
    )
    conn.commit()


def main():
    conn = init_db()
    session_id = start_session(conn)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        conn.close()
        raise RuntimeError("Could not open webcam.")

    model = YOLO("yolov8n.pt")  # downloads automatically on first run
    class_names = model.names

    last_log_time = 0.0

    try:
        while True:
            success, img = cap.read()
            if not success:
                print("Failed to read frame from webcam.")
                break

            results = model(img, stream=True, verbose=False)
            rows_to_log = []

            for r in results:
                for box in r.boxes:
                    confidence = float(box.conf[0])
                    if confidence < CONF_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    label = f"{class_names[cls]} {confidence:.2f}"

                    # Draw on the frame
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(img, label, (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    rows_to_log.append((
                        session_id,
                        datetime.now().isoformat(timespec="seconds"),
                        class_names[cls],
                        round(confidence, 2),
                        x1, y1, x2, y2,
                    ))

            # Write to the database at most once per LOG_INTERVAL
            now = time.time()
            if rows_to_log and now - last_log_time >= LOG_INTERVAL:
                conn.executemany(
                    """INSERT INTO detections
                       (session_id, detected_at, class_name, confidence,
                        x1, y1, x2, y2)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    rows_to_log,
                )
                conn.commit()
                last_log_time = now

            cv2.imshow("Webcam", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Always clean up, even if something crashes
        end_session(conn, session_id)
        conn.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
