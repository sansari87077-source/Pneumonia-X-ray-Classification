import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42


def count_split_images(split_dir: Path):
    counts = {}
    for cls in ["NORMAL", "PNEUMONIA"]:
        cls_dir = split_dir / cls
        if cls_dir.exists():
            counts[cls] = sum(
                1 for p in cls_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
        else:
            counts[cls] = 0
    return counts


def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def main():
    parser = argparse.ArgumentParser(description="Train DL-005 Pneumonia Classifier")
    parser.add_argument("--data_dir", type=str, default="data/chest_xray")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--fine_tune_epochs", type=int, default=0)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / args.data_dir
    models_dir = base_dir / "models"
    reports_dir = base_dir / "reports"

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    for folder in [train_dir, val_dir, test_dir]:
        if not folder.exists():
            raise FileNotFoundError(f"Missing dataset split folder: {folder}")

    class_balance = {
        "train": count_split_images(train_dir),
        "val": count_split_images(val_dir),
        "test": count_split_images(test_dir),
    }
    (reports_dir / "class_balance.json").write_text(
        json.dumps(class_balance, indent=2), encoding="utf-8"
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["NORMAL", "PNEUMONIA"],
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["NORMAL", "PNEUMONIA"],
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["NORMAL", "PNEUMONIA"],
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    file_paths = test_ds.file_paths

    model, base_model = build_model()
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    if args.fine_tune_epochs > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs)

    model_path = models_dir / "pneumonia_mobilenetv2.keras"
    model.save(model_path)

    # Evaluate on test set
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0).reshape(-1).astype(int)
    y_prob = model.predict(test_ds).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = float((tn + tp) / len(y_true))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall_sens = float(recall_score(y_true, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))

    metrics_payload = {
        "test_samples": int(len(y_true)),
        "accuracy": accuracy,
        "precision_pneumonia": precision,
        "recall_pneumonia_sensitivity": recall_sens,
        "specificity_normal": specificity,
        "f1_pneumonia": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        },
        "note": "Educational metrics only; not clinical validation.",
    }

    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )

    cls_report = classification_report(
        y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"], digits=4
    )
    (reports_dir / "classification_report.txt").write_text(cls_report, encoding="utf-8")

    with (reports_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["TN", "FP", "FN", "TP"])
        writer.writerow([int(tn), int(fp), int(fn), int(tp)])

    with (reports_dir / "false_negatives.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "true_label", "predicted_score"])
        for path, true_lbl, prob in zip(file_paths, y_true, y_prob):
            if true_lbl == 1 and prob < 0.5:
                writer.writerow([path, "PNEUMONIA", f"{prob:.4f}"])

    print(f"Training complete. Model saved to {model_path}")


if __name__ == "__main__":
    main()