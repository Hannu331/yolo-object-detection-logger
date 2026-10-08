# Real-Time Object Detection with Database Logging

A Python application that detects objects from a live webcam feed using **YOLOv8** and **OpenCV**, and stores every detection in a **SQLite** database. A separate report script uses **SQL queries** to analyze the stored data.

## Features

- Real-time object detection on webcam video (80 object classes from the COCO dataset)
- Bounding boxes with class name and confidence score drawn on the live feed
- Confidence threshold (0.5) to filter out weak detections
- Detections saved to SQLite, logged once per second to avoid duplicate flooding
- Session tracking: each run of the program is stored as a session with start and end time
- SQL-based analytics report (counts per class, average confidence, most detected object, recent detections)

## Tech Stack

| Area | Tool |
|------|------|
| Language | Python 3 |
| Object detection | YOLOv8 nano (Ultralytics), pretrained on COCO |
| Video capture and drawing | OpenCV |
| Database | SQLite (Python `sqlite3` module) |

## How It Works

```
Webcam -> OpenCV reads frame -> YOLOv8 detects objects -> Draw boxes on frame -> Show live window
                                        |
                                        v
                          Filter by confidence (>= 0.5)
                                        |
                                        v
                        Save to SQLite (once per second)
                                        |
                                        v
                       report.py runs SQL queries on the data
```

## Project Structure

```
yolo-object-detection-logger/
|-- main.py             # Detection + database logging
|-- report.py           # SQL analytics report
|-- requirements.txt    # Dependencies
|-- .gitignore
`-- README.md
```

## Database Schema

**sessions**

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER (PK) | Session ID |
| started_at | TEXT | Start time |
| ended_at | TEXT | End time |

**detections**

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER (PK) | Detection ID |
| session_id | INTEGER (FK -> sessions.id) | Which run it belongs to |
| detected_at | TEXT | Timestamp |
| class_name | TEXT | Detected object (e.g. person, laptop) |
| confidence | REAL | Model confidence (0 to 1) |
| x1, y1, x2, y2 | INTEGER | Bounding box coordinates |

One session has many detections (one-to-many relationship).

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/yolo-object-detection-logger.git
cd yolo-object-detection-logger

# 2. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the detector (press 'q' in the video window to quit)
python main.py

# View the analytics report
python report.py
```

The YOLOv8 weights (`yolov8n.pt`) download automatically on the first run.

## Sample Report Output

```
=== Detections per class ===
class | count | avg confidence
--------------------------------------------------
person | 2 | 0.9
cell phone | 1 | 0.77
```

## Screenshots

_Add a screenshot of the detection window here._

_Add a screenshot of the report output here._

## Limitations

- Uses a pretrained model, so it only recognizes the 80 COCO classes
- Smaller or distant objects and poor lighting reduce accuracy
- SQLite suits a single user; a multi-user version would need MySQL or PostgreSQL

## Future Improvements

- Web dashboard (Flask or Streamlit) to view reports in a browser
- Fine-tune the model on a custom dataset
- Object counting and alerts (for example, notify when more than N people appear)
- Export reports to CSV
