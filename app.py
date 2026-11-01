import csv
import json
import tempfile
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageOps

try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# =========================================================
# 1. PATHS & GLOBAL CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data" / "chest_xray"
PREDICTION_LOG_PATH = REPORTS_DIR / "prediction_log.csv"

MODEL_CANDIDATES = [
    MODELS_DIR / "pneumonia_custom_cnn.keras",
    MODELS_DIR / "pneumonia_mobilenetv2.keras",
]

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
THRESHOLD = 0.5
MAX_FILE_SIZE_MB = 5.0
ALLOWED_FORMATS = {"JPEG", "JPG", "PNG", "GIF"}

# =========================================================
# 2. SESSION STATE INITIALIZATION & PREDICTION LOGGING
# =========================================================


def load_persisted_predictions():
    if not PREDICTION_LOG_PATH.is_file():
        return []
    records = []
    try:
        with PREDICTION_LOG_PATH.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(
                    {
                        "timestamp": row.get("timestamp", ""),
                        "filename": row.get("filename", ""),
                        "model": row.get("model", ""),
                        "dimensions": row.get("dimensions", ""),
                        "label": row.get("label", ""),
                        "confidence": float(row.get("confidence", 0.0)),
                        "pneumonia_score": float(row.get("pneumonia_score", 0.0)),
                        "normal_score": float(row.get("normal_score", 0.0)),
                    }
                )
    except Exception:
        return []
    return records


def append_prediction_to_log(record: dict):
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        file_exists = PREDICTION_LOG_PATH.is_file()
        fieldnames = [
            "timestamp",
            "filename",
            "model",
            "dimensions",
            "label",
            "confidence",
            "pneumonia_score",
            "normal_score",
        ]
        with PREDICTION_LOG_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: record.get(k, "") for k in fieldnames})
    except Exception:
        pass


# =========================================================
# 3. STREAMLIT PAGE CONFIGURATION & DARK THEME CSS
# =========================================================

st.set_page_config(
    page_title="CheXplain Clinical Decision Support & X-Ray AI",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = load_persisted_predictions()
if "step_state" not in st.session_state:
    st.session_state["step_state"] = 1
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None


def render_markdown(body: str, **kwargs):
    if isinstance(body, str):
        body = textwrap.dedent(body).strip()
    return st.markdown(body, **kwargs)


render_markdown(
    """
<style>
.stApp {
    background-color: #0b0f17 !important;
    color: #f1f5f9 !important;
}
header[data-testid="stHeader"] {
    background-color: #0b0f17 !important;
    border-bottom: 1px solid #1a2232;
}
.block-container {
    max-width: 1560px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
section[data-testid="stSidebar"] {
    background-color: #0d121c !important;
    border-right: 1px solid #1a2232;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
.brand-wrapper {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 4px 18px 4px;
}
.brand-icon-box {
    width: 38px;
    height: 38px;
    background: #ffffff;
    color: #0b0f17 !important;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    font-weight: 900;
}
.nav-section-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    color: #64748b !important;
    text-transform: uppercase;
    margin-top: 16px;
    margin-bottom: 8px;
    padding-left: 4px;
}
section[data-testid="stSidebar"] [role="radiogroup"] label {
    background-color: #121824 !important;
    border: 1px solid #1e283b !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    width: 100% !important;
    transition: all 0.2s ease;
    margin-bottom: 6px;
}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background-color: #1a2336 !important;
    border-color: #10b981 !important;
}
.sidebar-profile-card {
    background: #121824;
    border: 1px solid #1e283b;
    border-radius: 12px;
    padding: 12px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 20px;
}
.doctor-badge {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: linear-gradient(135deg, #0284c7, #2563eb);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 13px;
    color: #ffffff !important;
}
.command-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
    background: #101622;
    border: 1px solid #1b2436;
    border-radius: 16px;
    padding: 18px 24px;
    margin-bottom: 22px;
}
.doctor-greeting {
    font-size: 24px;
    font-weight: 700;
    color: #f8fafc;
    display: flex;
    align-items: center;
    gap: 12px;
}
.live-pill {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #34d399;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.warn-pill {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.35);
    color: #fbbf24;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.search-pill {
    background: #151e2e;
    border: 1px solid #233148;
    color: #94a3b8;
    padding: 8px 16px;
    border-radius: 999px;
    font-size: 13px;
    display: inline-flex;
    align-items: center;
    gap: 10px;
}
.kpi-container {
    background: #111723;
    border: 1px solid #1c2638;
    border-radius: 16px;
    padding: 20px;
    height: 100%;
    position: relative;
}
.kpi-border-green { border-bottom: 3.5px solid #10b981; }
.kpi-border-orange { border-bottom: 3.5px solid #f97316; }
.kpi-border-amber { border-bottom: 3.5px solid #f59e0b; }
.kpi-border-purple { border-bottom: 3.5px solid #8b5cf6; }
.kpi-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.kpi-icon-pill {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}
.kpi-number {
    color: #f8fafc;
    font-size: 32px;
    font-weight: 800;
    line-height: 1.15;
    margin: 8px 0 6px 0;
}
.dashboard-panel {
    background: #111723;
    border: 1px solid #1c2638;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;
}
.panel-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
}
.panel-headline {
    font-size: 17px;
    font-weight: 700;
    color: #f8fafc;
}
.panel-subtext {
    font-size: 12px;
    color: #94a3b8;
    margin-bottom: 14px;
}
.patient-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
.patient-table th {
    text-align: left;
    color: #64748b;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    padding: 10px 8px;
    border-bottom: 1px solid #1e283b;
}
.patient-table td {
    padding: 12px 8px;
    border-bottom: 1px solid #161e2e;
    color: #e2e8f0;
    vertical-align: middle;
}
.badge-critical {
    background: rgba(239, 68, 68, 0.16);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.35);
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
}
.badge-monitoring {
    background: rgba(245, 158, 11, 0.16);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.35);
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
}
.badge-improving {
    background: rgba(16, 185, 129, 0.16);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
}
.chexplain-bar {
    background: #092540;
    border: 1px solid #143d66;
    color: #ffffff;
    padding: 14px 22px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 20px;
}
.stepper-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #111723;
    border: 1px solid #1c2638;
    border-radius: 12px;
    padding: 16px 24px;
    margin-bottom: 22px;
}
.step-item {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 14px;
    font-weight: 600;
    color: #64748b;
}
.step-item.active { color: #f8fafc; }
.step-item.done { color: #34d399; }
.circle-badge {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 700;
    border: 1.5px solid #475569;
    color: #64748b;
}
.step-item.active .circle-badge {
    background: #0ea5e9;
    border-color: #0ea5e9;
    color: #ffffff;
}
.step-item.done .circle-badge {
    background: #10b981;
    border-color: #10b981;
    color: #ffffff;
}
.step-divider-line {
    flex: 1;
    height: 1px;
    background: #233148;
    margin: 0 16px;
}
.val-item {
    display: flex;
    align-items: center;
    gap: 14px;
    background: #131a28;
    border: 1px solid #202c42;
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 10px;
}
.val-check-circle {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    border: 2px solid #22c55e;
    color: #22c55e;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 16px;
    flex-shrink: 0;
}
.val-warn-circle {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    border: 2px solid #f59e0b;
    color: #f59e0b;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 16px;
    flex-shrink: 0;
}
.val-error-circle {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    border: 2px solid #ef4444;
    color: #ef4444;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 16px;
    flex-shrink: 0;
}
.disclaimer-card {
    background: #111723;
    border: 1px solid #233148;
    border-left: 4px solid #ef4444;
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 26px;
    font-size: 12.5px;
    color: #94a3b8;
    line-height: 1.6;
}
[data-testid="stFileUploader"] {
    background: #111723 !important;
    border: 1.5px dashed #0284c7 !important;
    border-radius: 12px !important;
    padding: 18px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# 4. REAL REPORT & DATASET LOADERS
# =========================================================

def load_json_report(filename: str):
    path = REPORTS_DIR / filename
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def load_text_report(filename: str):
    path = REPORTS_DIR / filename
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None

def load_csv_rows(filename: str):
    path = REPORTS_DIR / filename
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []

def scan_data_directory_counts():
    if not DATA_DIR.is_dir():
        return None
    counts = {}
    found_any = False
    for split in ("train", "val", "test"):
        split_dir = DATA_DIR / split
        counts[split] = {"NORMAL": 0, "PNEUMONIA": 0}
        if split_dir.is_dir():
            for cls in ("NORMAL", "PNEUMONIA"):
                cls_dir = split_dir / cls
                if cls_dir.is_dir():
                    c = sum(
                        1
                        for p in cls_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                    )
                    counts[split][cls] = c
                    if c > 0:
                        found_any = True
    return counts if found_any else None

def load_dashboard_telemetry():
    metrics = load_json_report("metrics.json")
    cm_csv = load_csv_rows("confusion_matrix.csv")
    class_balance = load_json_report("class_balance.json") or scan_data_directory_counts()
    history = load_json_report("training_history.json")
    roc_data = load_json_report("roc_curve.json")
    # Accept legacy ROC files (fpr/tpr) as well as the dashboard schema.
    if isinstance(roc_data, dict):
        if "fpr_1" not in roc_data and "fpr" in roc_data:
            roc_data["fpr_1"] = roc_data["fpr"]
        if "tpr_1" not in roc_data and "tpr" in roc_data:
            roc_data["tpr_1"] = roc_data["tpr"]
        if not (
            isinstance(roc_data.get("fpr_1"), list)
            and isinstance(roc_data.get("tpr_1"), list)
            and len(roc_data["fpr_1"]) == len(roc_data["tpr_1"])
            and len(roc_data["fpr_1"]) >= 2
        ):
            roc_data = None
    false_negatives = load_csv_rows("false_negatives.csv")
    classification_report_txt = load_text_report("classification_report.txt")

    has_metrics = False
    tn = fp = fn = tp = None
    support_0 = support_1 = None
    prec_0 = rec_0 = f1_0 = None
    prec_1 = rec_1 = f1_1 = None
    acc = roc_auc = loss = val_loss = None
    epochs_trained = None

    if metrics is not None and "confusion_matrix" in metrics:
        has_metrics = True
        cm_dict = metrics["confusion_matrix"]
        tn = int(cm_dict.get("TN", 0))
        fp = int(cm_dict.get("FP", 0))
        fn = int(cm_dict.get("FN", 0))
        tp = int(cm_dict.get("TP", 0))
        support_0 = int(metrics.get("support_0", tn + fp))
        support_1 = int(metrics.get("support_1", fn + tp))
        prec_0 = float(metrics.get("prec_0", tn / (tn + fn) if (tn + fn) > 0 else 0.0))
        rec_0 = float(metrics.get("rec_0", tn / support_0 if support_0 > 0 else 0.0))
        f1_0 = float(
            metrics.get(
                "f1_0",
                2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0,
            )
        )
        prec_1 = float(metrics.get("prec_1", tp / (tp + fp) if (tp + fp) > 0 else 0.0))
        rec_1 = float(metrics.get("rec_1", tp / support_1 if support_1 > 0 else 0.0))
        f1_1 = float(
            metrics.get(
                "f1_1",
                2 * prec_1 * rec_1 / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0.0,
            )
        )
        total_eval = support_0 + support_1
        acc = float(metrics.get("accuracy", (tn + tp) / total_eval if total_eval > 0 else 0.0))
        roc_auc = float(metrics["roc_auc"]) if "roc_auc" in metrics else None
        loss = float(metrics["loss"]) if "loss" in metrics else None
        val_loss = float(metrics["val_loss"]) if "val_loss" in metrics else loss
        epochs_trained = metrics.get("epochs_trained")
    elif cm_csv:
        row = cm_csv[0]
        try:
            tn = int(row["TN"])
            fp = int(row["FP"])
            fn = int(row["FN"])
            tp = int(row["TP"])
            support_0 = tn + fp
            support_1 = fn + tp
            total_eval = support_0 + support_1
            prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            rec_0 = tn / support_0 if support_0 > 0 else 0.0
            f1_0 = 2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0
            prec_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec_1 = tp / support_1 if support_1 > 0 else 0.0
            f1_1 = 2 * prec_1 * rec_1 / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0.0
            acc = (tn + tp) / total_eval if total_eval > 0 else 0.0
            has_metrics = True
        except Exception:
            has_metrics = False

    total_dataset_scans = (
        sum(
            int(split.get("NORMAL", 0)) + int(split.get("PNEUMONIA", 0))
            for split in class_balance.values()
        )
        if class_balance
        else None
    )

    return {
        "has_metrics": has_metrics,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "support_0": support_0,
        "support_1": support_1,
        "prec_0": prec_0,
        "rec_0": rec_0,
        "f1_0": f1_0,
        "prec_1": prec_1,
        "rec_1": rec_1,
        "f1_1": f1_1,
        "accuracy": acc,
        "roc_auc": roc_auc,
        "loss": loss,
        "val_loss": val_loss,
        "epochs_trained": epochs_trained,
        "class_balance": class_balance,
        "total_dataset_scans": total_dataset_scans,
        "history": history,
        "roc_data": roc_data,
        "false_negatives": false_negatives,
        "classification_report_txt": classification_report_txt,
    }

# =========================================================
# 5. MODEL LOADING, VALIDATION, INFERENCE & GRAD-CAM
# =========================================================

@st.cache_resource(show_spinner="Loading trained Keras model...")
def _load_keras_model_cached(model_path_str: str, mtime: float):
    model = tf.keras.models.load_model(model_path_str, compile=False)
    if not hasattr(model, "predict"):
        raise TypeError(f"Loaded object from {model_path_str} is not a valid Keras model.")
    return model

def get_model():
    errors = []
    if not MODELS_DIR.exists():
        errors.append(f"Models directory does not exist: {MODELS_DIR}")
        return None, None, errors

    existing_candidates = [p for p in MODEL_CANDIDATES if p.is_file()]
    extra_keras_files = sorted(
        p for p in MODELS_DIR.glob("*.keras") if p not in existing_candidates
    )
    all_candidates = existing_candidates + extra_keras_files

    if not all_candidates:
        errors.append(
            "No `.keras` model file found in `models/`. Run `python train.py` to train and save a model."
        )
        return None, None, errors

    for candidate in all_candidates:
        try:
            if candidate.stat().st_size == 0:
                errors.append(f"{candidate.name}: Model file is empty (0 bytes).")
                continue
            mtime = candidate.stat().st_mtime
            model = _load_keras_model_cached(str(candidate), mtime)
            return model, candidate.name, errors
        except Exception as exc:
            errors.append(f"{candidate.name}: {type(exc).__name__} — {exc}")

    return None, None, errors

def get_model_expected_spec(model, model_name: str = ""):
    if model is not None and getattr(model, "input_shape", None) is not None:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        if len(shape) == 4:
            h = int(shape[1]) if shape[1] is not None else 150
            w = int(shape[2]) if shape[2] is not None else 150
            c = int(shape[3]) if shape[3] is not None else 1
            return h, w, c
    if "mobilenet" in str(model_name).lower():
        return 224, 224, 3
    return 150, 150, 1

def validate_uploaded_image(uploaded_file, model, model_name: str):
    checks = []
    is_valid = True

    # Validate file size
    size_bytes = getattr(uploaded_file, "size", 0)
    size_mb = size_bytes / (1024 * 1024)
    if 0 < size_mb <= MAX_FILE_SIZE_MB:
        checks.append(
            ("pass", "File size within limit", f"{size_mb:.2f} MB (Max: {MAX_FILE_SIZE_MB:.0f} MB)")
        )
    elif size_bytes == 0:
        checks.append(("error", "Empty file uploaded", "File size is 0 bytes"))
        return False, None, checks
    else:
        checks.append(
            (
                "error",
                "File size exceeds limit",
                f"{size_mb:.2f} MB exceeds {MAX_FILE_SIZE_MB:.0f} MB",
            )
        )
        return False, None, checks

    # Validate image format and reading
    try:
        uploaded_file.seek(0)
        raw_img = Image.open(uploaded_file)
        img_format = (raw_img.format or "UNKNOWN").upper()
        pil_img = ImageOps.exif_transpose(raw_img)
        pil_img.load()
    except Exception as exc:
        checks.append(("error", "Corrupted or unreadable image file", str(exc)))
        return False, None, checks

    if img_format in ALLOWED_FORMATS:
        checks.append(("pass", "Supported image format verified", f"Format: {img_format}"))
    else:
        checks.append(
            (
                "error",
                "Unsupported image format",
                f"Detected {img_format}; allowed: {', '.join(sorted(ALLOWED_FORMATS))}",
            )
        )
        is_valid = False

    # Validate dimensions
    w, h = pil_img.width, pil_img.height
    exp_h, exp_w, _ = get_model_expected_spec(model, model_name)
    min_dim = min(exp_h, exp_w)
    if w >= min_dim and h >= min_dim:
        checks.append(
            (
                "pass",
                "Image dimension requirements met",
                f"Current: {w} × {h} (Model target: {exp_w} × {exp_h})",
            )
        )
    elif w >= 128 and h >= 128:
        checks.append(
            (
                "warn",
                "Low-resolution image (will be interpolated)",
                f"Current: {w} × {h} (Recommended: ≥ {min_dim} × {min_dim})",
            )
        )
    else:
        checks.append(
            (
                "error",
                "Image resolution too small for diagnostic CNN",
                f"Current: {w} × {h} (Minimum required: 128 × 128)",
            )
        )
        is_valid = False

    aspect = w / h if h > 0 else 0.0
    if 0.55 <= aspect <= 1.85:
        checks.append(
            ("pass", "Thoracic aspect ratio within normal bounds", f"Width/Height ratio: {aspect:.2f}")
        )
    else:
        checks.append(
            (
                "error",
                "Extreme aspect ratio rejected",
                f"Ratio {aspect:.2f} outside [0.55, 1.85]",
            )
        )
        is_valid = False

    rgb_arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32)
    channel_diff = float(np.mean(np.max(rgb_arr, axis=-1) - np.min(rgb_arr, axis=-1)))
    if channel_diff <= 15.0:
        checks.append(
            (
                "pass",
                "Grayscale radiograph intensity profile verified",
                f"Mean RGB channel spread: {channel_diff:.1f}",
            )
        )
    else:
        checks.append(
            (
                "warn",
                "High color saturation detected — verify X-ray modality",
                f"Mean RGB channel spread: {channel_diff:.1f} (Expected near 0 for X-rays)",
            )
        )

    return is_valid, pil_img.convert("RGB"), checks

def preprocess_for_model(image: Image.Image, model, model_name: str = ""):
    exp_h, exp_w, exp_c = get_model_expected_spec(model, model_name)
    if exp_c == 1:
        processed = ImageOps.fit(
            image.convert("L"), (exp_w, exp_h), method=Image.Resampling.BILINEAR
        )
        tensor = np.asarray(processed, dtype=np.float32) / 255.0
        tensor = tensor[..., np.newaxis]
    else:
        processed = ImageOps.fit(
            image.convert("RGB"), (exp_w, exp_h), method=Image.Resampling.BILINEAR
        )
        tensor = np.asarray(processed, dtype=np.float32)
        has_internal_rescaling = any(
            isinstance(lyr, tf.keras.layers.Rescaling) for lyr in getattr(model, "layers", [])
        )
        if not has_internal_rescaling and "mobilenet" not in str(model_name).lower():
            tensor = tensor / 255.0

    batch_tensor = np.expand_dims(tensor, axis=0)
    return batch_tensor, processed

def run_inference(image: Image.Image, model, model_name: str):
    if model is None:
        raise RuntimeError(
            "No trained Keras model is loaded. Train a model with `python train.py` or place a valid `.keras` file in `models/`."
        )
    if image is None:
        raise ValueError("No input image was provided for inference.")

    tensor, processed_pil = preprocess_for_model(image, model, model_name)

    try:
        raw = model.predict(tensor, verbose=0)
    except Exception as exc:
        raise RuntimeError(f"Model forward pass failed: {exc}") from exc

    output = np.asarray(raw, dtype=np.float32).reshape(-1)
    if output.size not in (1, 2) or not np.all(np.isfinite(output)):
        raise ValueError(
            f"Invalid model output tensor: shape={np.asarray(raw).shape}, values={output}"
        )

    activation_name = getattr(getattr(model.layers[-1], "activation", None), "__name__", "")
    if output.size == 1:
        value = float(output[0])
        if activation_name == "sigmoid" or 0.0 <= value <= 1.0:
            pneumonia_prob = value
        elif activation_name in ("linear", "identity", ""):
            pneumonia_prob = float(1.0 / (1.0 + np.exp(-np.clip(value, -80, 80))))
        else:
            raise ValueError(f"Unsupported single-output activation: {activation_name}")
    else:
        if activation_name == "softmax" or np.isclose(np.sum(output), 1.0, atol=1e-2):
            probabilities = output
        elif activation_name in ("linear", "identity", ""):
            shifted = output - np.max(output)
            exp_values = np.exp(shifted)
            probabilities = exp_values / np.sum(exp_values)
        else:
            raise ValueError(f"Unsupported two-output activation: {activation_name}")
        pneumonia_prob = float(probabilities[1])

    pneumonia_prob = float(np.clip(pneumonia_prob, 0.0, 1.0))
    normal_prob = 1.0 - pneumonia_prob
    label = "PNEUMONIA" if pneumonia_prob >= THRESHOLD else "NORMAL"
    confidence = pneumonia_prob if label == "PNEUMONIA" else normal_prob

    return {
        "label": label,
        "confidence": confidence,
        "normal_score": normal_prob,
        "pneumonia_score": pneumonia_prob,
        "tensor": tensor,
        "processed_pil": processed_pil,
    }

def find_target_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return layer, sub.name
        if isinstance(layer, tf.keras.layers.Conv2D):
            return None, layer.name
    return None, None

def generate_gradcam_overlay(image: Image.Image, tensor: np.ndarray, model, alpha: float = 0.42):
    if model is None:
        return None, "No trained model loaded."
    try:
        nested_base, conv_name = find_target_conv_layer(model)
        if conv_name is None:
            return None, "No Conv2D layer found in model architecture."

        input_tensor = tf.cast(tensor, tf.float32)

        if nested_base is None:
            grad_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[model.get_layer(conv_name).output, model.output],
            )
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(input_tensor, training=False)
                if isinstance(predictions, (list, tuple)):
                    predictions = predictions[0]
                loss = predictions[:, 0] if predictions.shape[-1] == 1 else predictions[:, 1]
            grads = tape.gradient(loss, conv_outputs)
        else:
            sub_conv_model = tf.keras.Model(
                inputs=nested_base.inputs,
                outputs=[nested_base.get_layer(conv_name).output, nested_base.output],
            )
            head_layers = []
            passed_base = False
            for lyr in model.layers:
                if lyr.name == nested_base.name:
                    passed_base = True
                    continue
                if passed_base:
                    head_layers.append(lyr)

            with tf.GradientTape() as tape:
                x = input_tensor
                for lyr in model.layers:
                    if lyr.name == nested_base.name:
                        break
                    if not isinstance(lyr, tf.keras.layers.InputLayer):
                        x = lyr(x, training=False)
                conv_outputs, base_out = sub_conv_model(x, training=False)
                tape.watch(conv_outputs)
                y = base_out
                for lyr in head_layers:
                    y = lyr(y, training=False)
                loss = y[:, 0] if y.shape[-1] == 1 else y[:, 1]
            grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            return None, "Gradient computation returned None."

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_outputs[0]
        heatmap = tf.linalg.matmul(conv_out, pooled_grads[..., tf.newaxis])
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap_np = heatmap.numpy()

        jet = mpl.colormaps["jet"]
        jet_colors = jet(np.uint8(255 * heatmap_np))[:, :, :3]
        jet_pil = Image.fromarray(np.uint8(jet_colors * 255)).resize(
            (image.width, image.height), Image.Resampling.BILINEAR
        )

        base_rgb = image.convert("RGB")
        overlay_pil = Image.blend(base_rgb, jet_pil, alpha=alpha)
        return overlay_pil, conv_name
    except Exception as exc:
        return None, str(exc)

# =========================================================
# 6. PDF REPORT GENERATOR
# =========================================================

def build_pdf_report(record: dict, orig_img: Image.Image, gradcam_img: Image.Image = None) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_fill_color(9, 37, 64)
    pdf.rect(0, 0, 210, 32, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_xy(12, 8)
    pdf.cell(0, 8, "CheXplain AI - Chest X-Ray Analysis Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(12)
    pdf.cell(0, 6, f"Generated: {record['timestamp']} | Model Engine: {record['model']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(12)
    pdf.set_text_color(20, 25, 35)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "1. Diagnostic Screening Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(200, 210, 225)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(95, 7, f"File Name: {record['filename']}", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(95, 7, f"Original Resolution: {record['dimensions']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(95, 7, f"Predicted Finding: {record['label']}", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(
        95,
        7,
        f"Confidence Score: {record['confidence']*100:.2f}% (Threshold: {THRESHOLD})",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )
    pdf.cell(95, 7, f"Pneumonia Probability: {record['pneumonia_score']*100:.2f}%", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(95, 7, f"Normal Probability: {record['normal_score']*100:.2f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "2. Radiograph & Grad-CAM Localization", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    y_img = pdf.get_y()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_orig:
        orig_img.convert("RGB").resize((300, 300)).save(tmp_orig.name)
        pdf.image(tmp_orig.name, x=18, y=y_img, w=75)

    if gradcam_img is not None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_cam:
            gradcam_img.convert("RGB").resize((300, 300)).save(tmp_cam.name)
            pdf.image(tmp_cam.name, x=112, y=y_img, w=75)

    pdf.set_y(y_img + 78)
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(95, 5, "Original Frontal Radiograph", align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
    if gradcam_img is not None:
        pdf.cell(95, 5, "Grad-CAM Activation Overlay", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        pdf.ln(5)

    pdf.ln(8)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(185, 28, 28)
    pdf.cell(0, 7, "IMPORTANT NON-CLINICAL DISCLAIMER", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(60, 70, 85)
    disclaimer_text = (
        "This report is generated by an educational deep learning prototype (Ref X-Ray Exp / DL-005) "
        "and is strictly for academic and research demonstration purposes. Model scores are not calibrated "
        "clinical probabilities. Never use this output for clinical diagnosis, patient triage, or treatment "
        "decisions. Always consult a qualified radiologist or physician."
    )
    pdf.multi_cell(0, 5, disclaimer_text)

    return bytes(pdf.output())

# =========================================================
# 7. LOAD MODEL & REAL TELEMETRY
# =========================================================

loaded_model, active_model_file, model_load_errors = get_model()
telemetry = load_dashboard_telemetry()

# =========================================================
# 8. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    render_markdown(
        """
        <div class="brand-wrapper">
            <div class="brand-icon-box">🩻</div>
            <div>
                <div style="font-weight:800; font-size:16px; color:#f8fafc;">Pnuemonia X-ray Demo </div>
                <div style="font-size:11px; color:#94a3b8;">Pulmonology &amp; Radiology AI</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_markdown('<div class="nav-section-title">Overview</div>', unsafe_allow_html=True)
    selected_page = st.radio(
        "Main Navigation",
        [
            "Clinical Dashboard",
            "CheXplain Diagnostic Tool",
            "Model Evaluation Suite",
            "Dataset & Training Telemetry",
        ],
        label_visibility="collapsed",
    )

    session_pred_count = len(st.session_state["prediction_history"])
    dataset_scans_str = (
        f"{telemetry['total_dataset_scans']:,}"
        if telemetry["total_dataset_scans"] is not None
        else "N/A"
    )
    fn_alert_str = (
        str(len(telemetry["false_negatives"]))
        if telemetry["false_negatives"]
        else (str(telemetry["fn"]) if telemetry["fn"] is not None else "N/A")
    )

    render_markdown('<div class="nav-section-title">Live Telemetry</div>', unsafe_allow_html=True)
    render_markdown(
        f"""
        <div style="font-size:13px; color:#94a3b8; padding: 4px 6px; line-height: 2.1;">
            👥 Dataset Scans <span style="float:right; background:#0284c7; color:white; padding:1px 7px; border-radius:10px; font-size:11px; font-weight:700;">{dataset_scans_str}</span><br>
            ⚠️ Logged Misclassifications <span style="float:right; background:#f97316; color:white; padding:1px 7px; border-radius:10px; font-size:11px; font-weight:700;">{fn_alert_str}</span><br>
            🧬 Recorded Predictions <span style="float:right; background:#8b5cf6; color:white; padding:1px 7px; border-radius:10px; font-size:11px; font-weight:700;">{session_pred_count}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    render_markdown('<div class="nav-section-title">Inference Engine</div>', unsafe_allow_html=True)
    if loaded_model is not None:
        exp_h, exp_w, exp_c = get_model_expected_spec(loaded_model, active_model_file)
        st.success(f"Model Active: {active_model_file}")
        st.caption(f"Input Spec: {exp_w}×{exp_h}×{exp_c} | Params: {loaded_model.count_params():,}")
    else:
        st.error("Model unavailable — inference disabled.")
        for err_msg in model_load_errors:
            st.caption(f"• {err_msg}")

    render_markdown(
        """
        <div class="sidebar-profile-card">
            <div class="doctor-badge">AI</div>
            <div>
                <div style="font-weight:700; font-size:13.5px; color:#f8fafc;">Ref X-Ray Exp / DL-005</div>
                <div style="font-size:11.5px; color:#94a3b8;">Educational CNN Prototype</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# TOP HEADER BAR
# =========================================================

if telemetry["has_metrics"]:
    status_pill_html = '<span class="live-pill">● Live Evaluation Reports Connected</span>'
else:
    status_pill_html = '<span class="warn-pill">● Awaiting Evaluation Reports (Run train.py)</span>'

render_markdown(
    f"""
    <div class="command-header">
        <div>
            <div class="doctor-greeting">
                CheXplain Pneumonia Screening Workspace
                {status_pill_html}
            </div>
            <div style="font-size:13px; color:#94a3b8; margin-top:4px;">
                Thoracic &amp; Pulmonary Imaging • Ref X-Ray Exp / DL-005 Deep Learning System
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:12px;">
            <div class="search-pill">🔍 Logged Scans Evaluated: <b>{len(st.session_state['prediction_history'])}</b></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# PAGE 1: CLINICAL DASHBOARD
# =========================================================

if selected_page == "Clinical Dashboard":
    if not telemetry["has_metrics"]:
        st.warning(
            "No evaluation metrics found in `reports/metrics.json` or `reports/confusion_matrix.csv`. "
            "Run `python train.py` to evaluate your dataset and populate dashboard KPIs with real model telemetry."
        )

    has_m = telemetry["has_metrics"]
    total_eval = (telemetry["support_0"] + telemetry["support_1"]) if has_m else None
    total_errors = (telemetry["fp"] + telemetry["fn"]) if has_m else None

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        eval_str = f"{total_eval:,}" if total_eval is not None else "N/A"
        sub_c1 = (
            f"Normal: {telemetry['support_0']:,} • Pneumonia: {telemetry['support_1']:,}"
            if has_m
            else "Awaiting reports/metrics.json"
        )
        render_markdown(
            f"""
            <div class="kpi-container kpi-border-green">
                <div class="kpi-head">
                    <div class="kpi-icon-pill" style="background:rgba(16,185,129,0.15); color:#10b981;">🧑‍⚕️</div>
                    <span style="color:#10b981; font-size:12px; font-weight:700;">TEST COHORT</span>
                </div>
                <div style="color:#94a3b8; font-size:13px; font-weight:500; margin-top:6px;">Evaluated Cohort Scans</div>
                <div class="kpi-number">{eval_str}</div>
                <div style="color:#34d399; font-size:12px; font-weight:600;">{sub_c1}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        fn_str = f"{telemetry['fn']:,}" if has_m else "N/A"
        sub_c2 = (
            f"Sensitivity (Recall): {telemetry['rec_1']*100:.2f}%"
            if has_m
            else "Awaiting reports/metrics.json"
        )
        render_markdown(
            f"""
            <div class="kpi-container kpi-border-orange">
                <div class="kpi-head">
                    <div class="kpi-icon-pill" style="background:rgba(249,115,22,0.15); color:#f97316;">⚠️</div>
                    <span style="color:#f97316; font-size:12px; font-weight:700;">FALSE NEGATIVES</span>
                </div>
                <div style="color:#94a3b8; font-size:13px; font-weight:500; margin-top:6px;">Missed Pneumonia Cases</div>
                <div class="kpi-number">{fn_str}</div>
                <div style="color:#fb923c; font-size:12px; font-weight:600;">{sub_c2}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        acc_str = f"{telemetry['accuracy']*100:.2f}%" if has_m else "N/A"
        auc_part = f"ROC-AUC: {telemetry['roc_auc']:.4f}" if telemetry["roc_auc"] is not None else "ROC-AUC: N/A"
        loss_part = f"Loss: {telemetry['val_loss']:.4f}" if telemetry["val_loss"] is not None else "Loss: N/A"
        sub_c3 = f"{auc_part} • {loss_part}" if has_m else "Awaiting reports/metrics.json"
        render_markdown(
            f"""
            <div class="kpi-container kpi-border-amber">
                <div class="kpi-head">
                    <div class="kpi-icon-pill" style="background:rgba(245,158,11,0.15); color:#f59e0b;">🎯</div>
                    <span style="color:#f59e0b; font-size:12px; font-weight:700;">PERFORMANCE</span>
                </div>
                <div style="color:#94a3b8; font-size:13px; font-weight:500; margin-top:6px;">Model Accuracy Score</div>
                <div class="kpi-number">{acc_str}</div>
                <div style="color:#fbbf24; font-size:12px; font-weight:600;">{sub_c3}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        err_str = f"{total_errors:,}" if total_errors is not None else str(len(st.session_state["prediction_history"]))
        title_c4 = "Total Misclassifications (FP+FN)" if has_m else "Session Predictions Logged"
        sub_c4 = (
            f"FP: {telemetry['fp']} • Logged Runs: {len(st.session_state['prediction_history'])}"
            if has_m
            else f"Saved in reports/prediction_log.csv"
        )
        render_markdown(
            f"""
            <div class="kpi-container kpi-border-purple">
                <div class="kpi-head">
                    <div class="kpi-icon-pill" style="background:rgba(139,92,246,0.15); color:#8b5cf6;">✳️</div>
                    <span style="color:#8b5cf6; font-size:12px; font-weight:700;">AUDIT LOG</span>
                </div>
                <div style="color:#94a3b8; font-size:13px; font-weight:500; margin-top:6px;">{title_c4}</div>
                <div class="kpi-number">{err_str}</div>
                <div style="color:#a78bfa; font-size:12px; font-weight:600;">{sub_c4}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    mid_left, mid_right = st.columns([1.15, 1])

    with mid_left:
        render_markdown(
            """
            <div class="dashboard-panel">
                <div class="panel-header-row">
                    <div class="panel-headline">Dataset Partition Distribution</div>
                    <span style="font-size:12px; color:#94a3b8;">reports/class_balance.json</span>
                </div>
                <div class="panel-subtext">Actual radiograph counts across Train, Validation, and Test splits</div>
            """,
            unsafe_allow_html=True,
        )

        cb = telemetry["class_balance"]
        if cb is not None:
            splits = ["Train Split", "Val Split", "Test Split"]
            normal_counts = [
                int(cb.get("train", {}).get("NORMAL", 0)),
                int(cb.get("val", {}).get("NORMAL", 0)),
                int(cb.get("test", {}).get("NORMAL", 0)),
            ]
            pneu_counts = [
                int(cb.get("train", {}).get("PNEUMONIA", 0)),
                int(cb.get("val", {}).get("PNEUMONIA", 0)),
                int(cb.get("test", {}).get("PNEUMONIA", 0)),
            ]

            fig_splits = go.Figure()
            fig_splits.add_trace(
                go.Bar(
                    x=splits,
                    y=normal_counts,
                    name="NORMAL (Class 0)",
                    marker_color="#10b981",
                    text=normal_counts,
                    textposition="auto",
                )
            )
            fig_splits.add_trace(
                go.Bar(
                    x=splits,
                    y=pneu_counts,
                    name="PNEUMONIA (Class 1)",
                    marker_color="#3b82f6",
                    text=pneu_counts,
                    textposition="auto",
                )
            )
            fig_splits.update_layout(
                barmode="group",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=280,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                font=dict(color="#94a3b8"),
            )
            st.plotly_chart(fig_splits, use_container_width=True)
        else:
            st.info(
                "No `reports/class_balance.json` or `data/chest_xray` directory found. "
                "Place your dataset in `data/chest_xray` or run `python train.py`."
            )
        render_markdown("</div>", unsafe_allow_html=True)

    with mid_right:
        render_markdown(
            """
            <div class="dashboard-panel">
                <div class="panel-header-row">
                    <div class="panel-headline">Diagnostic Outcome Mix</div>
                    <span style="background:#151e2e; border:1px solid #243148; padding:3px 10px; border-radius:999px; font-size:11px; color:#94a3b8;">reports/metrics.json</span>
                </div>
                <div class="panel-subtext">Breakdown of True Positives, True Negatives, False Positives &amp; False Negatives</div>
            """,
            unsafe_allow_html=True,
        )

        if has_m:
            fig_donut = go.Figure(
                data=[
                    go.Pie(
                        labels=[
                            "True Positive (Pneumonia)",
                            "True Negative (Normal)",
                            "False Negative (Missed Pneumonia)",
                            "False Positive (Normal Flagged)",
                        ],
                        values=[telemetry["tp"], telemetry["tn"], telemetry["fn"], telemetry["fp"]],
                        hole=0.68,
                        marker=dict(colors=["#3b82f6", "#10b981", "#ef4444", "#f59e0b"]),
                        textinfo="none",
                    )
                ]
            )
            fig_donut.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=215,
                margin=dict(l=10, r=10, t=5, b=5),
                annotations=[
                    dict(
                        text=f"<b>{total_eval:,}</b><br><span style='font-size:11px;color:#94a3b8'>evaluated</span>",
                        x=0.5,
                        y=0.5,
                        font_size=18,
                        showarrow=False,
                        font_color="#f8fafc",
                    )
                ],
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

            pneu_ratio = (telemetry["support_1"] / max(total_eval, 1)) * 100
            render_markdown(
                f"""
                <div style="background:#131c2a; border:1px solid #1e2a3e; border-radius:10px; padding:10px 14px; font-size:12px; color:#94a3b8; line-height:1.5;">
                    💡 <b>Class Balance Telemetry:</b> Pneumonia cases represent <b>{pneu_ratio:.1f}%</b> of evaluated scans.
                    Sensitivity: <b>{telemetry['rec_1']*100:.1f}%</b> • Specificity: <b>{telemetry['rec_0']*100:.1f}%</b>.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Run `python train.py` to generate confusion matrix outcomes in `reports/metrics.json`.")
        render_markdown("</div>", unsafe_allow_html=True)

    bot_left, bot_right = st.columns([1.15, 1])

    with bot_left:
        render_markdown(
            """
            <div class="dashboard-panel">
                <div class="panel-header-row">
                    <div class="panel-headline">CNN Training &amp; Validation Telemetry Across Epochs</div>
                    <span style="background:#151e2e; border:1px solid #243148; padding:3px 10px; border-radius:999px; font-size:11px; color:#94a3b8;">reports/training_history.json</span>
                </div>
                <div class="panel-subtext">Real epoch-by-epoch accuracy progression recorded during training</div>
            """,
            unsafe_allow_html=True,
        )

        hist = telemetry["history"]
        if hist and "accuracy" in hist and "val_accuracy" in hist:
            ep_axis = [f"Ep {i+1}" for i in range(len(hist["accuracy"]))]
            fig_telemetry = go.Figure()
            fig_telemetry.add_trace(
                go.Scatter(
                    x=ep_axis,
                    y=[v * 100 for v in hist["accuracy"]],
                    name="Train Accuracy (%)",
                    line=dict(color="#10b981", width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(16, 185, 129, 0.10)",
                )
            )
            fig_telemetry.add_trace(
                go.Scatter(
                    x=ep_axis,
                    y=[v * 100 for v in hist["val_accuracy"]],
                    name="Val Accuracy (%)",
                    line=dict(color="#3b82f6", width=2.5, dash="dot"),
                )
            )
            fig_telemetry.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=290,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(title="Accuracy (%)"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
                font=dict(color="#94a3b8"),
            )
            st.plotly_chart(fig_telemetry, use_container_width=True)
        else:
            st.info("No `reports/training_history.json` found. Run `python train.py` to log training curves.")
        render_markdown("</div>", unsafe_allow_html=True)

    with bot_right:
        render_markdown(
            """
            <div class="dashboard-panel">
                <div class="panel-header-row">
                    <div class="panel-headline">Real Inference &amp; Misclassification Audit Log</div>
                    <span style="background:#151e2e; border:1px solid #243148; padding:3px 10px; border-radius:999px; font-size:11px; color:#94a3b8;">Live Data</span>
                </div>
                <div class="panel-subtext">Recent session predictions and logged test set misclassifications</div>
            """,
            unsafe_allow_html=True,
        )

        rows_html = []
        for rec in reversed(st.session_state["prediction_history"][-6:]):
            badge_cls = "badge-critical" if rec["label"] == "PNEUMONIA" else "badge-improving"
            rows_html.append(
                f"<tr>"
                f"<td><b>{rec['filename']}</b><br><span style='font-size:11px;color:#64748b;'>{rec['timestamp']}</span></td>"
                f"<td>Session Inference</td>"
                f"<td><span class='{badge_cls}'>{rec['label']}</span></td>"
                f"<td>{rec['pneumonia_score']*100:.1f}%</td>"
                f"</tr>"
            )

        if not rows_html and telemetry["false_negatives"]:
            for fn_row in telemetry["false_negatives"][:6]:
                fname = Path(fn_row.get("file_path", "scan")).name
                true_l = fn_row.get("true_label", "")
                pred_l = fn_row.get("predicted_label", "")
                score_val = float(fn_row.get("pneumonia_score", 0.0)) * 100
                badge_cls = "badge-critical" if true_l == "PNEUMONIA" else "badge-monitoring"
                rows_html.append(
                    f"<tr>"
                    f"<td><b>{fname}</b></td>"
                    f"<td>True: {true_l}</td>"
                    f"<td><span class='{badge_cls}'>Pred: {pred_l}</span></td>"
                    f"<td>{score_val:.1f}%</td>"
                    f"</tr>"
                )

        if rows_html:
            table_html = (
                "<table class='patient-table'>"
                "<thead><tr><th>Scan File</th><th>Source / Ground Truth</th><th>Outcome</th><th>Pneumonia Score</th></tr></thead>"
                "<tbody>" + "".join(rows_html) + "</tbody></table>"
            )
            render_markdown(table_html, unsafe_allow_html=True)
        else:
            st.info(
                "No session predictions or misclassification logs recorded yet. "
                "Evaluate an X-ray in the CheXplain Diagnostic Tool or run `python train.py`."
            )
        render_markdown("</div>", unsafe_allow_html=True)

# =========================================================
# PAGE 2: CHEXPLAIN DIAGNOSTIC TOOL
# =========================================================

elif selected_page == "CheXplain Diagnostic Tool":
    render_markdown(
        '<div class="chexplain-bar">🩻 CheXplain Diagnostic Screening &amp; Grad-CAM Localization</div>',
        unsafe_allow_html=True,
    )

    if loaded_model is None:
        st.error(
            "🚫 **Trained Model Unavailable — Inference Disabled.**\n\n"
            "No valid `.keras` model could be loaded from `models/`. Fallback/heuristic predictions have been "
            "disabled to ensure all outputs come strictly from a trained neural network."
        )
        if model_load_errors:
            with st.expander("View Model Load Diagnostics", expanded=True):
                for err in model_load_errors:
                    st.write(f"- `{err}`")
        st.info("Train a model with `python train.py` and reload the app.")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload Frontal Chest Radiograph (JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a frontal AP/PA chest X-ray image (max 5 MB).",
    )

    if uploaded_file is None:
        st.session_state["step_state"] = 1
        st.session_state["last_result"] = None

    s1_cls = "done" if uploaded_file is not None else "active"
    s2_cls = (
        "done"
        if st.session_state["last_result"] is not None
        else ("active" if uploaded_file is not None else "")
    )
    s3_cls = "active" if st.session_state["last_result"] is not None else ""

    render_markdown(
        f"""
        <div class="stepper-container">
            <div class="step-item {s1_cls}">
                <div class="circle-badge">1</div>
                <span>Upload &amp; Validate Radiograph</span>
            </div>
            <div class="step-divider-line"></div>
            <div class="step-item {s2_cls}">
                <div class="circle-badge">2</div>
                <span>CNN Forward Inference</span>
            </div>
            <div class="step-divider-line"></div>
            <div class="step-item {s3_cls}">
                <div class="circle-badge">3</div>
                <span>Grad-CAM &amp; Report Export</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if uploaded_file is not None:
        is_valid, pil_img, checks = validate_uploaded_image(
            uploaded_file, loaded_model, active_model_file
        )

        col_val, col_preview = st.columns([1.15, 1])

        with col_val:
            render_markdown(
                '<div class="dashboard-panel"><div class="panel-headline">Pre-Inference Quality &amp; Format Checks</div>',
                unsafe_allow_html=True,
            )
            for status, title, detail in checks:
                icon_cls = (
                    "val-check-circle"
                    if status == "pass"
                    else ("val-warn-circle" if status == "warn" else "val-error-circle")
                )
                symbol = "✓" if status == "pass" else ("!" if status == "warn" else "✕")
                render_markdown(
                    f"""
                    <div class="val-item">
                        <div class="{icon_cls}">{symbol}</div>
                        <div>
                            <div style="font-weight:700; font-size:13.5px; color:#f8fafc;">{title}</div>
                            <div style="font-size:12px; color:#94a3b8;">{detail}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            render_markdown("</div>", unsafe_allow_html=True)

        with col_preview:
            render_markdown(
                '<div class="dashboard-panel"><div class="panel-headline">Radiograph Loupe &amp; Contrast Controls</div>',
                unsafe_allow_html=True,
            )
            if pil_img is not None:
                c_ctrl1, c_ctrl2 = st.columns(2)
                with c_ctrl1:
                    contrast_factor = st.slider("Contrast", 0.5, 2.5, 1.0, 0.05)
                with c_ctrl2:
                    brightness_factor = st.slider("Brightness", 0.5, 2.0, 1.0, 0.05)

                disp_img = ImageEnhance.Contrast(pil_img).enhance(contrast_factor)
                disp_img = ImageEnhance.Brightness(disp_img).enhance(brightness_factor)
                st.image(disp_img, caption=f"{uploaded_file.name} ({pil_img.width}×{pil_img.height})", width="stretch")
            render_markdown("</div>", unsafe_allow_html=True)

        if not is_valid:
            st.error("Uploaded file failed validation checks. Please upload a valid chest X-ray image.")
        else:
            if st.button("🚀 Run Trained CNN Inference & Grad-CAM", type="primary", width="stretch"):
                with st.spinner(f"Running forward pass on {active_model_file}..."):
                    try:
                        result = run_inference(pil_img, loaded_model, active_model_file)
                        gradcam_pil, conv_layer_used = generate_gradcam_overlay(
                            pil_img, result["tensor"], loaded_model
                        )
                        record = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "filename": uploaded_file.name,
                            "model": active_model_file,
                            "dimensions": f"{pil_img.width}x{pil_img.height}",
                            "label": result["label"],
                            "confidence": result["confidence"],
                            "pneumonia_score": result["pneumonia_score"],
                            "normal_score": result["normal_score"],
                        }
                        st.session_state["prediction_history"].append(record)
                        append_prediction_to_log(record)
                        st.session_state["last_result"] = {
                            "record": record,
                            "gradcam_pil": gradcam_pil,
                            "conv_layer": conv_layer_used,
                            "orig_pil": pil_img,
                        }
                    except Exception as exc:
                        st.error(f"Inference failed: {exc}")
                        st.session_state["last_result"] = None

        if st.session_state["last_result"] is not None:
            res_bundle = st.session_state["last_result"]
            rec = res_bundle["record"]
            gradcam_pil = res_bundle["gradcam_pil"]
            conv_layer = res_bundle["conv_layer"]

            st.divider()
            r_col1, r_col2 = st.columns([1.05, 1])

            with r_col1:
                is_pneu = rec["label"] == "PNEUMONIA"
                accent = "#ef4444" if is_pneu else "#10b981"
                render_markdown(
                    f"""
                    <div class="dashboard-panel" style="border-left: 5px solid {accent};">
                        <div style="font-size:12px; color:#94a3b8; text-transform:uppercase; font-weight:700;">
                            Model Classification Output ({rec['model']})
                        </div>
                        <div style="font-size:32px; font-weight:800; color:{accent}; margin:6px 0;">
                            {rec['label']}
                        </div>
                        <div style="font-size:14px; color:#e2e8f0; margin-bottom:14px;">
                            Confidence: <b>{rec['confidence']*100:.2f}%</b> &nbsp;|&nbsp; Decision Threshold: <b>{THRESHOLD}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                fig_probs = go.Figure(
                    go.Bar(
                        x=[rec["normal_score"] * 100, rec["pneumonia_score"] * 100],
                        y=["NORMAL (Class 0)", "PNEUMONIA (Class 1)"],
                        orientation="h",
                        marker_color=["#10b981", "#ef4444"],
                        text=[f"{rec['normal_score']*100:.2f}%", f"{rec['pneumonia_score']*100:.2f}%"],
                        textposition="auto",
                    )
                )
                fig_probs.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=180,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(range=[0, 100], title="Probability (%)"),
                )
                st.plotly_chart(fig_probs, use_container_width=True)

                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        "📥 Export JSON Result",
                        data=json.dumps(rec, indent=2),
                        file_name=f"prediction_{Path(rec['filename']).stem}.json",
                        mime="application/json",
                        width="stretch",
                    )
                with dl_col2:
                    if FPDF_AVAILABLE:
                        pdf_bytes = build_pdf_report(rec, res_bundle["orig_pil"], gradcam_pil)
                        st.download_button(
                            "📄 Export PDF Report",
                            data=pdf_bytes,
                            file_name=f"CheXplain_Report_{Path(rec['filename']).stem}.pdf",
                            mime="application/pdf",
                            width="stretch",
                        )

            with r_col2:
                render_markdown(
                    f"""
                    <div class="dashboard-panel">
                        <div class="panel-headline">Grad-CAM Spatial Activation Overlay</div>
                        <div class="panel-subtext">Extracted from layer: <code>{conv_layer}</code></div>
                    """,
                    unsafe_allow_html=True,
                )
                if gradcam_pil is not None:
                    st.image(gradcam_pil, caption="Grad-CAM Class Activation Map", width="stretch")
                else:
                    st.warning(f"Grad-CAM unavailable: {conv_layer}")
                render_markdown("</div>", unsafe_allow_html=True)

# =========================================================
# PAGE 3: MODEL EVALUATION SUITE
# =========================================================

elif selected_page == "Model Evaluation Suite":
    if not telemetry["has_metrics"]:
        st.warning(
            "No evaluation artifacts found in `reports/`. Run `python train.py` to generate `metrics.json`, "
            "`roc_curve.json`, `classification_report.txt`, and `false_negatives.csv`."
        )
    else:
        ev_left, ev_right = st.columns(2)

        with ev_left:
            render_markdown(
                """
                <div class="dashboard-panel">
                    <div class="panel-headline">Confusion Matrix (Held-Out Test Cohort)</div>
                    <div class="panel-subtext">Loaded from reports/metrics.json</div>
                """,
                unsafe_allow_html=True,
            )
            cm_matrix = [
                [telemetry["tn"], telemetry["fp"]],
                [telemetry["fn"], telemetry["tp"]],
            ]
            fig_cm = go.Figure(
                data=go.Heatmap(
                    z=cm_matrix,
                    x=["Predicted NORMAL", "Predicted PNEUMONIA"],
                    y=["Actual NORMAL", "Actual PNEUMONIA"],
                    text=[[f"TN: {telemetry['tn']}", f"FP: {telemetry['fp']}"], [f"FN: {telemetry['fn']}", f"TP: {telemetry['tp']}"]],
                    texttemplate="%{text}",
                    textfont={"size": 16, "color": "#ffffff"},
                    colorscale="Blues",
                    showscale=False,
                )
            )
            fig_cm.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=310,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            render_markdown("</div>", unsafe_allow_html=True)

        with ev_right:
            render_markdown(
                """
                <div class="dashboard-panel">
                    <div class="panel-headline">Receiver Operating Characteristic (ROC) Curve</div>
                    <div class="panel-subtext">Held-out test-set ROC • reports/roc_curve.json</div>
                """,
                unsafe_allow_html=True,
            )
            roc_d = telemetry["roc_data"]
            if roc_d and "fpr_1" in roc_d and "tpr_1" in roc_d:
                fig_roc = go.Figure()
                fig_roc.add_trace(
                    go.Scatter(
                        x=roc_d["fpr_1"],
                        y=roc_d["tpr_1"],
                        name=f"PNEUMONIA (AUC = {roc_d.get('roc_auc', telemetry['roc_auc']):.4f})",
                        line=dict(color="#3b82f6", width=2.5),
                    )
                )
                if "fpr_0" in roc_d and "tpr_0" in roc_d:
                    fig_roc.add_trace(
                        go.Scatter(
                            x=roc_d["fpr_0"],
                            y=roc_d["tpr_0"],
                            name="NORMAL (Class 0)",
                            line=dict(color="#10b981", width=2, dash="dot"),
                        )
                    )
                fig_roc.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        name="Random Baseline",
                        line=dict(color="#64748b", width=1.5, dash="dash"),
                    )
                )
                fig_roc.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=310,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(title="False Positive Rate"),
                    yaxis=dict(title="True Positive Rate"),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("No valid ROC data found in reports/roc_curve.json. Run `python evaluate.py` to calculate the curve from the held-out test images.")
            render_markdown("</div>", unsafe_allow_html=True)

        render_markdown(
            f"""
            <div class="dashboard-panel">
                <div class="panel-headline">Per-Class Diagnostic Performance Table</div>
                <div class="panel-subtext">Real precision, recall, F1-score, and support from held-out evaluation</div>
                <table class="patient-table">
                    <thead>
                        <tr>
                            <th>Class Label</th>
                            <th>Precision</th>
                            <th>Recall (Sensitivity / Specificity)</th>
                            <th>F1-Score</th>
                            <th>Support (Scans)</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><b>NORMAL (Class 0)</b></td>
                            <td>{telemetry['prec_0']*100:.2f}%</td>
                            <td>{telemetry['rec_0']*100:.2f}% (Specificity)</td>
                            <td>{telemetry['f1_0']*100:.2f}%</td>
                            <td>{telemetry['support_0']:,}</td>
                        </tr>
                        <tr>
                            <td><b>PNEUMONIA (Class 1)</b></td>
                            <td>{telemetry['prec_1']*100:.2f}%</td>
                            <td>{telemetry['rec_1']*100:.2f}% (Sensitivity)</td>
                            <td>{telemetry['f1_1']*100:.2f}%</td>
                            <td>{telemetry['support_1']:,}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if telemetry["classification_report_txt"]:
            with st.expander("View Raw Scikit-Learn Classification Report (`reports/classification_report.txt`)"):
                st.code(telemetry["classification_report_txt"], language="text")

        if telemetry["false_negatives"]:
            with st.expander(f"Inspect Logged Misclassified Scans ({len(telemetry['false_negatives'])} cases in `reports/false_negatives.csv`)"):
                st.dataframe(telemetry["false_negatives"], use_container_width=True)

# =========================================================
# PAGE 4: DATASET & TRAINING TELEMETRY
# =========================================================

elif selected_page == "Dataset & Training Telemetry":
    t_col1, t_col2 = st.columns(2)

    with t_col1:
        render_markdown(
            """
            <div class="dashboard-panel">
                <div class="panel-headline">Training &amp; Validation Loss Curve</div>
                <div class="panel-subtext">Loaded from reports/training_history.json</div>
            """,
            unsafe_allow_html=True,
        )
        hist = telemetry["history"]
        if hist and "loss" in hist and "val_loss" in hist:
            ep_axis = [f"Ep {i+1}" for i in range(len(hist["loss"]))]
            fig_loss = go.Figure()
            fig_loss.add_trace(
                go.Scatter(
                    x=ep_axis,
                    y=hist["loss"],
                    name="Train Loss",
                    line=dict(color="#f59e0b", width=2.5),
                )
            )
            fig_loss.add_trace(
                go.Scatter(
                    x=ep_axis,
                    y=hist["val_loss"],
                    name="Validation Loss",
                    line=dict(color="#ef4444", width=2.5, dash="dot"),
                )
            )
            fig_loss.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(title="Binary Crossentropy Loss"),
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.info("No `reports/training_history.json` available.")
        render_markdown("</div>", unsafe_allow_html=True)

    with t_col2:
        render_markdown(
            """
            <div class="dashboard-panel">
                <div class="panel-headline">Learning Rate Schedule (`ReduceLROnPlateau`)</div>
                <div class="panel-subtext">Loaded from reports/training_history.json</div>
            """,
            unsafe_allow_html=True,
        )
        if hist and hist.get("lr"):
            ep_axis = [f"Ep {i+1}" for i in range(len(hist["lr"]))]
            fig_lr = go.Figure(
                go.Scatter(
                    x=ep_axis,
                    y=hist["lr"],
                    name="Learning Rate",
                    line=dict(color="#8b5cf6", width=2.5),
                    mode="lines+markers",
                )
            )
            fig_lr.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(title="Learning Rate", type="log"),
            )
            st.plotly_chart(fig_lr, use_container_width=True)
        else:
            st.info("Learning rate history not found in `reports/training_history.json`.")
        render_markdown("</div>", unsafe_allow_html=True)

    render_markdown(
        """
        <div class="dashboard-panel">
            <div class="panel-headline">Loaded Keras Model Architecture Summary</div>
            <div class="panel-subtext">Live inspection of layers, output shapes, and trainable parameters</div>
        """,
        unsafe_allow_html=True,
    )
    if loaded_model is not None:
        layer_rows = []
        for lyr in loaded_model.layers:
            layer_rows.append(
                {
                    "Layer Name": lyr.name,
                    "Layer Type": lyr.__class__.__name__,
                    "Output Shape": str(getattr(lyr, "output_shape", "N/A")),
                    "Parameters": f"{lyr.count_params():,}",
                }
            )
        st.dataframe(layer_rows, use_container_width=True)
    else:
        st.warning("No model currently loaded in `models/`.")
    render_markdown("</div>", unsafe_allow_html=True)

# =========================================================
# CLINICAL DISCLAIMER FOOTER
# =========================================================

render_markdown(
    """
    <div class="disclaimer-card">
        <b style="color:#f87171;">⚠️ EDUCATIONAL &amp; RESEARCH USE ONLY — NOT A MEDICAL DEVICE:</b>
        CheXplain (Ref X-Ray Exp / DL-005) is an academic deep learning demonstration built to study convolutional neural networks,
        dataset class imbalance, and Grad-CAM interpretability on chest radiographs. Predictions and confidence scores must never be used
        for clinical diagnosis, patient triage, or treatment decisions.
    </div>
    """,
    unsafe_allow_html=True,
)
