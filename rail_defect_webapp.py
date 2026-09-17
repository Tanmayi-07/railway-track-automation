import os
from typing import List, Tuple, Dict

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


# ---- Configuration ----

# Update this path if your best model is saved elsewhere
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "runs",
    "detect",
    "train",
    "weights",
    "best.pt",
)


# Map model class names to severity levels and suggestions (all your classes)
SEVERITY_CONFIG: Dict[str, Dict[str, str]] = {
    "Cracks": {
        "severity": "High",
        "color": "#ff4b5c",
        "suggestion": (
            "Critical rail crack detected. Immediately arrange detailed ultrasonic testing, "
            "enforce speed restrictions, and prepare for urgent rail replacement to avoid fracture."
        ),
    },
    "Flakings": {
        "severity": "Medium",
        "color": "#ffb347",
        "suggestion": (
            "Surface flaking observed. Monitor this area for material loss, schedule grinding or "
            "profiling, and include it in the next preventive maintenance cycle."
        ),
    },
    "Grooves": {
        "severity": "Medium",
        "color": "#ffb347",
        "suggestion": (
            "Groove-type wear detected on the rail head. Check wheel–rail contact conditions, "
            "plan corrective grinding, and monitor for further deepening."
        ),
    },
    "Joints": {
        "severity": "Low",
        "color": "#3ac569",
        "suggestion": (
            "Irregularity around a rail joint detected. Log this location, verify fasteners and "
            "joint gaps during routine inspection, and tighten or realign if required."
        ),
    },
    "Putus": {
        "severity": "High",
        "color": "#ff4b5c",
        "suggestion": (
            "Severe discontinuity/putus indication. Treat this as a potentially broken or highly "
            "compromised section. Stop traffic if confirmed, perform emergency inspection, and "
            "replace the affected rail immediately."
        ),
    },
    "Shellings": {
        "severity": "High",
        "color": "#ff4b5c",
        "suggestion": (
            "Severe shelling detected. Immediately schedule on-site inspection, apply temporary "
            "speed restrictions, and plan partial or full rail replacement."
        ),
    },
    "Spallings": {
        "severity": "Medium",
        "color": "#ffb347",
        "suggestion": (
            "Spalling observed. Monitor the affected region closely, schedule maintenance in the "
            "next inspection window, and verify fastening and ballast conditions."
        ),
    },
    "Squats": {
        "severity": "Medium",
        "color": "#ffb347",
        "suggestion": (
            "Squat defect detected. Plan rail grinding or localized repair, monitor for growth, "
            "and assess wheel and track alignment to reduce further damage."
        ),
    },
}

DEFAULT_DEFECT_CONFIG = {
    "severity": "Medium",
    "color": "#4fa9ff",
    "suggestion": (
        "Rail anomaly detected. Review this segment during the next inspection and validate with "
        "detailed measurements or non-destructive testing."
    ),
}


def load_model(model_path: str = DEFAULT_MODEL_PATH) -> YOLO:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"YOLO model not found at '{model_path}'. "
            "Update DEFAULT_MODEL_PATH in 'rail_defect_webapp.py' to your best.pt path."
        )
    model = YOLO(model_path)
    return model


model = load_model()
CLASS_NAMES = model.names


def get_defect_config(label_name: str) -> Dict[str, str]:
    return SEVERITY_CONFIG.get(label_name, DEFAULT_DEFECT_CONFIG)


def summarize_overall_severity(severity_levels: List[str]) -> str:
    if not severity_levels:
        return "No defects detected. Track section appears healthy. Continue routine monitoring."

    if "High" in severity_levels:
        return (
            "Overall condition: **CRITICAL**.\n\n"
            "- Immediately schedule detailed inspection.\n"
            "- Enforce temporary speed restrictions over this section.\n"
            "- Prioritize rail repair or replacement."
        )
    if "Medium" in severity_levels:
        return (
            "Overall condition: **WARNING**.\n\n"
            "- Plan maintenance in the upcoming inspection window.\n"
            "- Monitor crack propagation and repeat scans after some operation cycles."
        )
    return (
        "Overall condition: **STABLE**.\n\n"
        "- Track is usable but should remain under regular surveillance.\n"
        "- Re-scan this section periodically to ensure no rapid degradation."
    )


def format_suggestions_table(
    records: List[Tuple[str, float, str, str]],
) -> str:
    """
    records: list of (defect_name, confidence, severity, suggestion)
    """
    if not records:
        return "### Suggestions\n\nNo cracks or rail defects were detected in this image."

    header = "| Defect | Confidence | Severity | Suggested Action |\n"
    header += "|--------|------------|----------|------------------|\n"

    rows = []
    for name, conf, severity, suggestion in records:
        rows.append(
            f"| {name} | {conf:.2f} | **{severity}** | {suggestion} |"
        )

    return "### Detected Defects & Actions\n\n" + header + "\n".join(rows)


def predict(image: Image.Image):
    image = image.convert("RGB")
    results = model(image)
    boxes = results[0].boxes

    pred_boxes = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4))
    pred_scores = boxes.conf.cpu().numpy() if boxes.conf is not None else np.array([])
    pred_labels = (
        boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.array([])
    )

    draw = ImageDraw.Draw(image)

    # Optional: try loading a nicer font; fallback to default if not available
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        pass

    records: List[Tuple[str, float, str, str]] = []
    severity_levels: List[str] = []

    for box, label_idx, score in zip(pred_boxes, pred_labels, pred_scores):
        x_min, y_min, x_max, y_max = box
        label_name = CLASS_NAMES.get(int(label_idx), f"Class {label_idx}")
        cfg = get_defect_config(label_name)

        severity = cfg["severity"]
        color = cfg["color"]
        suggestion = cfg["suggestion"]

        severity_levels.append(severity)
        records.append((label_name, float(score), severity, suggestion))

        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=4)

        text = f"{label_name} ({severity}) - {score:.2f}"
        text_position = (x_min + 4, max(y_min - 24, 0))

        # Draw text with simple background for readability
        if font is not None:
            text_size = draw.textbbox((0, 0), text, font=font)
            bw = text_size[2] - text_size[0] + 8
            bh = text_size[3] - text_size[1] + 4
            draw.rectangle(
                [text_position, (text_position[0] + bw, text_position[1] + bh)],
                fill="rgba(0, 0, 0, 160)",
            )
            draw.text(text_position, text, fill="white", font=font)
        else:
            draw.text(text_position, text, fill=color)

    suggestions_md = format_suggestions_table(records)
    overall_md = summarize_overall_severity(severity_levels)

    full_markdown = (
        "## Rail Safety Assessment\n\n"
        + overall_md
        + "\n\n"
        + suggestions_md
    )

    return image, full_markdown


CUSTOM_CSS = """
body {
    background: radial-gradient(circle at top left, #0b1829, #020308);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.rail-header {
    background: linear-gradient(135deg, #133b5c, #1e5f74, #3a7ca5);
    border-radius: 18px;
    padding: 18px 24px;
    color: #f4f8fb;
    box-shadow: 0 18px 45px rgba(0, 0, 0, 0.45);
    display: flex;
    align-items: center;
    gap: 18px;
    position: relative;
    overflow: hidden;
}
.rail-header::before {
    content: "";
    position: absolute;
    inset: -40%;
    background: radial-gradient(circle at 10% 0, rgba(255,255,255,0.12), transparent 55%);
    opacity: 0.7;
    animation: drift 12s linear infinite;
}
.rail-header-content {
    position: relative;
    z-index: 1;
}
.rail-title {
    font-size: 1.7rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}
.rail-subtitle {
    font-size: 0.95rem;
    opacity: 0.92;
}
.rail-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 999px;
    padding: 4px 10px;
    background: rgba(4, 220, 167, 0.12);
    color: #b1ffe5;
    font-size: 0.8rem;
    border: 1px solid rgba(4, 220, 167, 0.4);
}
.rail-glow-orbit {
    width: 68px;
    height: 68px;
    border-radius: 999px;
    border: 2px solid rgba(255,255,255,0.18);
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
}
.rail-glow-dot {
    width: 40px;
    height: 40px;
    border-radius: 999px;
    background: conic-gradient(from 200deg, #04dca7, #4fa9ff, #ffb347, #04dca7);
    animation: pulse 3s ease-in-out infinite;
    box-shadow: 0 0 22px rgba(79, 169, 255, 0.75);
}
.rail-card {
    background: rgba(10, 18, 33, 0.96);
    border-radius: 16px;
    padding: 16px 18px;
    border: 1px solid rgba(134, 174, 220, 0.24);
    box-shadow: 0 14px 32px rgba(0, 0, 0, 0.4);
}
.rail-card h3 {
    margin-top: 0;
    color: #f4f8fb;
}
.rail-metric {
    font-size: 0.9rem;
    color: #a6b4ca;
}
.rail-footer {
    font-size: 0.8rem;
    color: #6b7b93;
    margin-top: 4px;
}

@keyframes pulse {
    0%, 100% { transform: scale(0.94); box-shadow: 0 0 14px rgba(79, 169, 255, 0.6); }
    50% { transform: scale(1); box-shadow: 0 0 26px rgba(4, 220, 167, 0.85); }
}
@keyframes drift {
    0% { transform: translateX(-8%); opacity: 0.7; }
    50% { transform: translateX(8%); opacity: 0.4; }
    100% { transform: translateX(-8%); opacity: 0.7; }
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="Rail Crack & Defect Detection",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan"),
    ) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML(
                    """
                    <div class="rail-header">
                        <div class="rail-glow-orbit">
                            <div class="rail-glow-dot"></div>
                        </div>
                        <div class="rail-header-content">
                            <div class="rail-pill">
                                Live rail health assistant
                            </div>
                            <div class="rail-title">
                                Intelligent Rail Crack & Defect Detection
                            </div>
                            <div class="rail-subtitle">
                                Upload a track image to detect cracks, classify their severity,
                                and receive railway safety suggestions in real time.
                            </div>
                        </div>
                    </div>
                    """
                )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload rail track image",
                    type="pil",
                    height=420,
                )

                gr.Markdown(
                    "Upload a clear image focusing on the rail head and web region "
                    "for the most accurate crack detection."
                )

            with gr.Column(scale=1.1):
                with gr.Group(elem_classes="rail-card"):
                    output_image = gr.Image(
                        label="Detected defects & severity",
                        height=420,
                    )
                    suggestions = gr.Markdown(
                        value="Safety summary and maintenance suggestions will appear here "
                        "after you upload an image.",
                    )
                gr.Markdown(
                    '<div class="rail-footer">Model: YOLO-based rail defect detector · '
                    "Severity rules are configurable inside the application code.</div>"
                )

        input_image.change(
            fn=predict,
            inputs=input_image,
            outputs=[output_image, suggestions],
        )

        return demo


app = build_app()


if __name__ == "__main__":
    app.launch()

