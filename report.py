# Reads detections.db and prints summary reports using SQL.
# Run after you've used yolo_webcam_db.py at least once:  python report.py

import sqlite3

DB_PATH = "detections.db"


def print_table(title, headers, rows):
    print(f"\n=== {title} ===")
    if not rows:
        print("(no data)")
        return
    print(" | ".join(headers))
    print("-" * 50)
    for row in rows:
        print(" | ".join(str(v) for v in row))


def main():
    conn = sqlite3.connect(DB_PATH)

    # 1. Sessions with their detection counts (LEFT JOIN + GROUP BY)
    rows = conn.execute(
        """
        SELECT s.id, s.started_at, s.ended_at, COUNT(d.id) AS total
        FROM sessions s
        LEFT JOIN detections d ON d.session_id = s.id
        GROUP BY s.id
        ORDER BY s.id DESC
        """
    ).fetchall()
    print_table("Sessions", ["id", "started", "ended", "detections"], rows)

    # 2. Count and average confidence per object class
    rows = conn.execute(
        """
        SELECT class_name, COUNT(*) AS total, ROUND(AVG(confidence), 2) AS avg_conf
        FROM detections
        GROUP BY class_name
        ORDER BY total DESC
        """
    ).fetchall()
    print_table("Detections per class", ["class", "count", "avg confidence"], rows)

    # 3. Most frequently detected object
    rows = conn.execute(
        """
        SELECT class_name, COUNT(*) AS total
        FROM detections
        GROUP BY class_name
        ORDER BY total DESC
        LIMIT 1
        """
    ).fetchall()
    print_table("Most detected object", ["class", "count"], rows)

    # 4. Last 10 detections
    rows = conn.execute(
        """
        SELECT detected_at, class_name, confidence
        FROM detections
        ORDER BY id DESC
        LIMIT 10
        """
    ).fetchall()
    print_table("Last 10 detections", ["time", "class", "confidence"], rows)

    conn.close()


if __name__ == "__main__":
    main()
