import os
from flask import Flask, request, render_template, redirect, url_for
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Setup Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLO model (ensure best.pt is in the same directory or update path)
model = YOLO("best.pt")
CLASS_NAMES = model.names

# Severity configuration based on defect type (all your classes, with speed limits)
SEVERITY_CONFIG = {
    "Cracks": {
        "severity": "High",
        "color": "#ff4b5c",
        "speed_kmph": 20,
        "suggestion": (
            "Critical rail crack detected. Immediately arrange detailed ultrasonic testing, "
            "impose a temporary speed limit of at most 20 km/h over this section, and prepare for "
            "urgent rail replacement to avoid fracture."
        ),
    },
    "Flakings": {
        "severity": "Medium",
        "color": "#ffb347",
        "speed_kmph": 40,
        "suggestion": (
            "Surface flaking observed. Monitor this area for material loss, schedule grinding or "
            "profiling, and temporarily limit speed to around 40 km/h until the rail head is "
            "restored."
        ),
    },
    "Grooves": {
        "severity": "Medium",
        "color": "#ffb347",
        "speed_kmph": 40,
        "suggestion": (
            "Groove-type wear detected on the rail head. Check wheel–rail contact conditions, "
            "plan corrective grinding, and operate trains at no more than 40 km/h on this segment "
            "until corrected."
        ),
    },
    "Joints": {
        "severity": "Low",
        "color": "#3ac569",
        "speed_kmph": 60,
        "suggestion": (
            "Irregularity around a rail joint detected. Log this location, verify fasteners and "
            "joint gaps during routine inspection, and, if necessary, reduce speed locally to "
            "around 60 km/h until the joint is adjusted."
        ),
    },
    "Putus": {
        "severity": "High",
        "color": "#ff4b5c",
        "speed_kmph": 10,
        "suggestion": (
            "Severe discontinuity/putus indication. Treat this as a potentially broken or highly "
            "compromised section. Stop traffic if confirmed, or operate at walking speed "
            "(≤ 10 km/h) only for emergency moves until the rail is replaced."
        ),
    },
    "Shellings": {
        "severity": "High",
        "color": "#ff4b5c",
        "speed_kmph": 20,
        "suggestion": (
            "Severe shelling detected. Immediately schedule on-site inspection, apply a temporary "
            "speed limit of at most 20 km/h, and plan partial or full rail replacement."
        ),
    },
    "Spallings": {
        "severity": "Medium",
        "color": "#ffb347",
        "speed_kmph": 40,
        "suggestion": (
            "Spalling observed. Monitor the affected region closely, schedule maintenance in the "
            "next inspection window, and temporarily restrict speed to around 40 km/h."
        ),
    },
    "Squats": {
        "severity": "Medium",
        "color": "#ffb347",
        "speed_kmph": 40,
        "suggestion": (
            "Squat defect detected. Plan rail grinding or localized repair, monitor for growth, "
            "and limit speed to around 40 km/h on this section until repaired."
        ),
    },
}

DEFAULT_DEFECT_CONFIG = {
    "severity": "Medium",
    "color": "#4fa9ff",
    "speed_kmph": 60,
    "suggestion": (
        "Rail anomaly detected. Include this segment in the next inspection round, validate with "
        "detailed measurements or non-destructive testing, and consider a local speed limit of "
        "around 60 km/h until cleared."
    ),
}


def get_defect_config(label_name: str):
    return SEVERITY_CONFIG.get(label_name, DEFAULT_DEFECT_CONFIG)


def summarize_overall_severity(severity_levels, speeds_kmph):
    if not severity_levels:
        return {
            "headline": "Track section appears healthy.",
            "detail": (
                "No cracks or visible rail defects were detected in this image. "
                "Continue routine monitoring and periodic inspections."
            ),
            "badge": "No Defects Detected",
            "badge_class": "badge bg-success",
            "speed_text": "Recommended speed: normal line speed as per route standard.",
        }

    min_speed = None
    valid_speeds = [s for s in (speeds_kmph or []) if isinstance(s, (int, float))]
    if valid_speeds:
        min_speed = min(valid_speeds)

    if "High" in severity_levels:
        speed_text = (
            f"Recommended temporary speed limit over this segment: ≤ {min_speed:.0f} km/h "
            "until detailed inspection and repair are completed."
            if min_speed is not None
            else "Apply a very low temporary speed limit (e.g. 10–20 km/h) until inspected."
        )
        return {
            "headline": "Overall condition: CRITICAL",
            "detail": (
                "High-severity defects detected. Immediately plan detailed on-site inspection, "
                "enforce temporary speed restrictions over this section, and prioritize rail repair "
                "or replacement."
            ),
            "badge": "Critical",
            "badge_class": "badge bg-danger severity-badge-high",
            "speed_text": speed_text,
        }

    if "Medium" in severity_levels:
        speed_text = (
            f"Recommended temporary speed limit over this segment: ≤ {min_speed:.0f} km/h "
            "until maintenance is completed."
            if min_speed is not None
            else "Apply a moderate temporary speed limit (around 40 km/h) until maintenance."
        )
        return {
            "headline": "Overall condition: WARNING",
            "detail": (
                "Medium-severity defects detected. Schedule maintenance in the upcoming inspection "
                "window and monitor crack propagation with follow-up scans."
            ),
            "badge": "Warning",
            "badge_class": "badge bg-warning text-dark severity-badge-medium",
            "speed_text": speed_text,
        }

    speed_text = (
        f"Suggested local speed limit: ≤ {min_speed:.0f} km/h until the low-severity issues "
        "are checked and confirmed stable."
        if min_speed is not None
        else "Suggested local speed limit: around 60 km/h with continued monitoring."
    )
    return {
        "headline": "Overall condition: STABLE",
        "detail": (
            "Only low-severity issues detected. Track is usable but should remain under regular "
            "surveillance and periodic re-scanning."
        ),
        "badge": "Stable",
        "badge_class": "badge bg-info severity-badge-low",
        "speed_text": speed_text,
    }


# Prediction function
def detect_defects(image_path):
    image = Image.open(image_path).convert("RGB")
    results = model(image)
    pred_boxes = results[0].boxes.xyxy.cpu().numpy()
    pred_scores = results[0].boxes.conf.cpu().numpy()
    pred_labels = results[0].boxes.cls.cpu().numpy().astype(int)

    draw = ImageDraw.Draw(image)

    # Try a nicer font if available
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        pass

    detections = []
    severity_levels = []
    speeds_kmph = []

    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        x_min, y_min, x_max, y_max = box
        label_name = CLASS_NAMES.get(label, f"Class {label}")
        cfg = get_defect_config(label_name)

        severity = cfg["severity"]
        color = cfg["color"]
        suggestion = cfg["suggestion"]
        speed_kmph = cfg.get("speed_kmph")

        severity_levels.append(severity)
        speeds_kmph.append(speed_kmph)
        detections.append(
            {
                "name": label_name,
                "confidence": float(score),
                "severity": severity,
                "suggestion": suggestion,
                "speed_kmph": speed_kmph,
            }
        )

        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=4)
        text = f"{label_name} ({severity}) - {score:.2f}"
        text_x, text_y = x_min + 4, max(y_min - 24, 0)

        if font is not None:
            bbox = draw.textbbox((0, 0), text, font=font)
            bw = bbox[2] - bbox[0] + 8
            bh = bbox[3] - bbox[1] + 4
            draw.rectangle(
                [(text_x, text_y), (text_x + bw, text_y + bh)],
                fill=(0, 0, 0, 160),
            )
            draw.text((text_x, text_y), text, fill="white", font=font)
        else:
            draw.text((text_x, text_y), text, fill=color)

    summary = summarize_overall_severity(severity_levels, speeds_kmph)

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.png')
    image.save(output_path)
    return output_path, detections, summary

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)
            output_path, detections, summary = detect_defects(image_path)
            return render_template(
                "predict.html",
                original=file.filename,
                output='output.png',
                detections=detections,
                summary=summary,
            )
    return render_template("predict.html", original=None, output=None, detections=None, summary=None)

# Start the server
if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
