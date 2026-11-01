
"""
Evaluate the same Keras model and preprocessing used by app.py.

Creates reports/roc_curve.json with the field names expected by app.py.
Evaluates the held-out test folder only.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import roc_auc_score, roc_curve


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
TEST_DIR = BASE_DIR / "data" / "chest_xray" / "test"
REPORTS_DIR = BASE_DIR / "reports"

# Keep this order identical to MODEL_CANDIDATES in app.py.
MODEL_CANDIDATES = [
    MODELS_DIR / "pneumonia_custom_cnn.keras",
    MODELS_DIR / "pneumonia_mobilenetv2.keras",
]

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
BATCH_SIZE = 16
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_active_model():
    """Load the first valid model in the same order as app.py."""

    errors = []

    for model_path in MODEL_CANDIDATES:

        if not model_path.is_file():
            continue

        try:
            print(f"Loading model: {model_path.name}")

            model = tf.keras.models.load_model(
                model_path,
                compile=False
            )

            if not hasattr(model, "predict"):
                raise TypeError(
                    "Loaded object does not provide predict()."
                )

            return model, model_path.name

        except Exception as exc:
            errors.append(
                f"{model_path.name}: {type(exc).__name__}: {exc}"
            )

    details = "\n".join(errors) if errors else (
        "No candidate .keras model files were found."
    )

    raise FileNotFoundError(
        f"Could not load a model from {MODELS_DIR}.\n{details}"
    )


def get_model_spec(model):

    shape = model.input_shape

    if isinstance(shape, list):
        shape = shape[0]

    if not shape or len(shape) != 4:
        raise ValueError(
            f"Expected a 4D image input, got model input shape: {shape}"
        )

    height = int(shape[1]) if shape[1] is not None else 224
    width = int(shape[2]) if shape[2] is not None else 224
    channels = int(shape[3]) if shape[3] is not None else 3

    if channels not in (1, 3):
        raise ValueError(
            f"Unsupported input channels: {channels}"
        )

    return height, width, channels


def preprocess_image(
    image_path: Path,
    model,
    model_name: str,
    image_size,
    channels
):
    """Preprocess image using the app.py preprocessing rules."""

    height, width = image_size

    with Image.open(image_path) as opened:

        image = ImageOps.exif_transpose(opened)
        image.load()

        if channels == 1:

            processed = ImageOps.fit(
                image.convert("L"),
                (width, height),
                method=Image.Resampling.BILINEAR
            )

            array = np.asarray(
                processed,
                dtype=np.float32
            ) / 255.0

            array = array[..., np.newaxis]

        else:

            processed = ImageOps.fit(
                image.convert("RGB"),
                (width, height),
                method=Image.Resampling.BILINEAR
            )

            array = np.asarray(
                processed,
                dtype=np.float32
            )

            has_internal_rescaling = any(
                isinstance(layer, tf.keras.layers.Rescaling)
                for layer in getattr(model, "layers", [])
            )

            if (
                not has_internal_rescaling
                and "mobilenet" not in model_name.lower()
            ):
                array = array / 255.0

    return array


def pneumonia_probabilities(raw_predictions, model):
    """Convert model output into P(PNEUMONIA)."""

    raw = np.asarray(
        raw_predictions,
        dtype=np.float32
    )

    if raw.ndim != 2 or raw.shape[1] not in (1, 2):
        raise ValueError(
            f"Unsupported model output shape: {raw.shape}"
        )

    if not np.all(np.isfinite(raw)):
        raise ValueError(
            "Model returned NaN or infinite predictions."
        )

    activation_name = getattr(
        getattr(model.layers[-1], "activation", None),
        "__name__",
        ""
    )

    if raw.shape[1] == 1:

        values = raw[:, 0]

        if (
            activation_name == "sigmoid"
            or np.all((values >= 0.0) & (values <= 1.0))
        ):
            probabilities = values

        elif activation_name in ("linear", "identity", ""):

            probabilities = 1.0 / (
                1.0 + np.exp(-np.clip(values, -80, 80))
            )

        else:
            raise ValueError(
                f"Unsupported single-output activation: {activation_name}"
            )

    else:

        if (
            activation_name == "softmax"
            or np.allclose(
                np.sum(raw, axis=1),
                1.0,
                atol=1e-2
            )
        ):
            probabilities = raw[:, 1]

        elif activation_name in ("linear", "identity", ""):

            shifted = raw - np.max(
                raw,
                axis=1,
                keepdims=True
            )

            exp_values = np.exp(shifted)

            softmax_values = exp_values / np.sum(
                exp_values,
                axis=1,
                keepdims=True
            )

            probabilities = softmax_values[:, 1]

        else:
            raise ValueError(
                f"Unsupported two-output activation: {activation_name}"
            )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "Converted probabilities contain invalid values."
        )

    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError(
            "Model outputs could not be interpreted as probabilities."
        )

    return probabilities


def collect_test_files():

    if not TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"Test dataset folder not found: {TEST_DIR}"
        )

    paths = []
    labels = []

    for class_index, class_name in enumerate(CLASS_NAMES):

        class_dir = TEST_DIR / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing test class folder: {class_dir}"
            )

        class_files = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not class_files:
            raise ValueError(
                f"No JPG/JPEG/PNG images found in {class_dir}"
            )

        paths.extend(class_files)
        labels.extend([class_index] * len(class_files))

    return paths, np.asarray(labels, dtype=np.int32)


def main():

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model, model_name = load_active_model()

    height, width, channels = get_model_spec(model)

    print(f"Active model: {model_name}")
    print(f"Expected input: {width} x {height} x {channels}")

    image_paths, y_true = collect_test_files()

    print(f"Found {len(image_paths)} test images.")

    print(
        f"NORMAL: {int(np.sum(y_true == 0))} | "
        f"PNEUMONIA: {int(np.sum(y_true == 1))}"
    )

    probability_batches = []

    for start in range(0, len(image_paths), BATCH_SIZE):

        batch_paths = image_paths[
            start:start + BATCH_SIZE
        ]

        batch_arrays = [
            preprocess_image(
                path,
                model,
                model_name,
                (height, width),
                channels
            )
            for path in batch_paths
        ]

        batch = np.stack(
            batch_arrays,
            axis=0
        ).astype(np.float32)

        raw_predictions = model.predict(
            batch,
            verbose=0
        )

        probability_batches.append(
            pneumonia_probabilities(
                raw_predictions,
                model
            )
        )

        print(
            f"Evaluated "
            f"{min(start + len(batch_paths), len(image_paths))}"
            f"/{len(image_paths)} images"
        )

    y_prob = np.concatenate(probability_batches)

    if len(y_true) != len(y_prob):
        raise ValueError(
            f"Label/prediction count mismatch: "
            f"{len(y_true)} vs {len(y_prob)}"
        )

    if len(np.unique(y_true)) != 2:
        raise ValueError(
            "ROC calculation requires both NORMAL and PNEUMONIA images."
        )

    # Positive class = PNEUMONIA (class index 1).
    fpr_1, tpr_1, thresholds_1 = roc_curve(
        y_true,
        y_prob,
        pos_label=1
    )

    roc_auc = float(
        roc_auc_score(y_true, y_prob)
    )

    # One-vs-rest curve for NORMAL.
    fpr_0, tpr_0, thresholds_0 = roc_curve(
        1 - y_true,
        1.0 - y_prob,
        pos_label=1
    )

    roc_data = {

        "fpr_1": fpr_1.tolist(),
        "tpr_1": tpr_1.tolist(),
        "thresholds_1": thresholds_1.tolist(),

        "fpr_0": fpr_0.tolist(),
        "tpr_0": tpr_0.tolist(),
        "thresholds_0": thresholds_0.tolist(),

        "fpr": fpr_1.tolist(),
        "tpr": tpr_1.tolist(),

        "roc_auc": roc_auc,

        "positive_class": "PNEUMONIA",
        "class_names": CLASS_NAMES,
        "model": model_name,

        "test_images": int(len(image_paths)),
        "normal_images": int(np.sum(y_true == 0)),
        "pneumonia_images": int(np.sum(y_true == 1))
    }

    output_path = REPORTS_DIR / "roc_curve.json"

    output_path.write_text(
        json.dumps(roc_data, indent=4),
        encoding="utf-8"
    )

    print("\nROC evaluation completed.")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Saved: {output_path}")
    print(f"Evaluated model: {model_name}")


if __name__ == "__main__":
    main()
