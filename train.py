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
    roc_curve,
)

# Parameters matching PDF Slides 13, 14, 21, 24
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

class LRLogger(tf.keras.callbacks.Callback):
    """Custom callback to force Keras to log the learning rate every epoch."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Safely extract the learning rate from the optimizer
        lr = getattr(self.model.optimizer, "learning_rate", getattr(self.model.optimizer, "lr", None))
        if lr is not None:
            if hasattr(lr, 'numpy'):
                logs['lr'] = float(lr.numpy())
            else:
                logs['lr'] = float(tf.keras.backend.get_value(lr))

def build_custom_cnn(input_shape=(150, 150, 1)):
    """
    Custom 5-Block Deep CNN matching PDF Slide 23:
    Conv2D -> BatchNormalization -> MaxPooling2D -> Dropout -> Flatten -> Dense
    """
    inputs = tf.keras.Input(shape=input_shape, name="xray_input")

    # Block 1
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv_block1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn_block1")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool_block1")(x)

    # Block 2
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv_block2")(x)
    x = tf.keras.layers.Dropout(0.1, name="drop_block2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_block2")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool_block2")(x)

    # Block 3
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv_block3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_block3")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool_block3")(x)

    # Block 4
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv_block4")(x)
    x = tf.keras.layers.Dropout(0.2, name="drop_block4")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_block4")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool_block4")(x)

    # Block 5 (Named 'last_conv_layer' for Grad-CAM extraction)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="last_conv_layer")(x)
    x = tf.keras.layers.Dropout(0.2, name="drop_block5")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_block5")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool_block5")(x)

    # Dense Classification Head
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    x = tf.keras.layers.Dropout(0.2, name="drop_head")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="pneumonia_prob")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Pneumonia_Custom_CNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_data_augmentation_layer():
    """
    Data Augmentation matching Slide 21:
    Rotation 30 deg, Zoom 20%, Shift 10%, Horizontal Flip
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.083),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomFlip("horizontal"),
        ],
        name="augmentation_pipeline",
    )


def count_split_images(split_dir: Path):
    counts = {}
    for cls in ["NORMAL", "PNEUMONIA"]:
        cls_dir = split_dir / cls
        counts[cls] = (
            sum(1 for p in cls_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if cls_dir.exists()
            else 0
        )
    return counts


def main():
    parser = argparse.ArgumentParser(description="Train Custom CNN for Pneumonia Detection")
    parser.add_argument("--data_dir", type=str, default="data/chest_xray")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
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
            raise FileNotFoundError(f"Missing dataset directory: {folder}")

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
        color_mode="grayscale",
        class_names=["NORMAL", "PNEUMONIA"],
        image_size=IMG_SIZE,
        batch_size=args.batch_size,
        shuffle=True,
        seed=SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="binary",
        color_mode="grayscale",
        class_names=["NORMAL", "PNEUMONIA"],
        image_size=IMG_SIZE,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="binary",
        color_mode="grayscale",
        class_names=["NORMAL", "PNEUMONIA"],
        image_size=IMG_SIZE,
        batch_size=args.batch_size,
        shuffle=False,
    )

    test_file_paths = test_ds.file_paths

    rescale = tf.keras.layers.Rescaling(1.0 / 255.0)
    aug = get_data_augmentation_layer()

    train_ds_proc = train_ds.map(lambda x, y: (aug(rescale(x), training=True), y)).prefetch(tf.data.AUTOTUNE)
    val_ds_proc = val_ds.map(lambda x, y: (rescale(x), y)).prefetch(tf.data.AUTOTUNE)
    test_ds_proc = test_ds.map(lambda x, y: (rescale(x), y)).prefetch(tf.data.AUTOTUNE)

    model = build_custom_cnn(input_shape=(150, 150, 1))

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        patience=2,
        factor=0.3,
        min_lr=1e-5,
        verbose=1,
    )
    
    # Initialize the custom learning rate logger
    lr_logger = LRLogger()

    print(f"\nStarting training for {args.epochs} epochs...")
    history = model.fit(
        train_ds_proc,
        validation_data=val_ds_proc,
        epochs=args.epochs,
        callbacks=[reduce_lr, lr_logger], # Register the logger here
    )

    model_path = models_dir / "pneumonia_custom_cnn.keras"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Evaluate on held-out test split
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0).reshape(-1).astype(int)
    y_prob = model.predict(test_ds_proc).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total_samples = int(len(y_true))
    acc = float((tn + tp) / total_samples) if total_samples > 0 else 0.0

    # Per-class metrics (0 = NORMAL, 1 = PNEUMONIA)
    prec_0 = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    rec_0 = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_0 = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    prec_1 = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec_1 = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    roc_auc = float(roc_auc_score(y_true, y_prob))
    fpr_1, tpr_1, _ = roc_curve(y_true, y_prob, pos_label=1)
    fpr_0, tpr_0, _ = roc_curve(1 - y_true, 1.0 - y_prob, pos_label=1)

    metrics_payload = {
        "test_samples": total_samples,
        "accuracy": acc,
        "loss": float(history.history["loss"][-1]),
        "val_loss": float(history.history["val_loss"][-1]),
        "roc_auc": roc_auc,
        "prec_0": prec_0,
        "rec_0": rec_0,
        "f1_0": f1_0,
        "support_0": int(tn + fp),
        "prec_1": prec_1,
        "rec_1": rec_1,
        "f1_1": f1_1,
        "support_1": int(fn + tp),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "epochs_trained": args.epochs,
    }
    (reports_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    # Save exact ROC curve coordinates for Panel (b)
    roc_payload = {
        "roc_auc": roc_auc,
        "fpr_1": [float(x) for x in fpr_1],
        "tpr_1": [float(y) for y in tpr_1],
        "fpr_0": [float(x) for x in fpr_0],
        "tpr_0": [float(y) for y in tpr_0],
    }
    (reports_dir / "roc_curve.json").write_text(json.dumps(roc_payload, indent=2), encoding="utf-8")

    # Save training history safely parsing the custom lr log
    lr_hist = history.history.get("lr", history.history.get("learning_rate", []))
    history_dict = {
        "accuracy": [float(v) for v in history.history["accuracy"]],
        "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
        "loss": [float(v) for v in history.history["loss"]],
        "val_loss": [float(v) for v in history.history["val_loss"]],
        "lr": [float(v) for v in lr_hist],
    }
    (reports_dir / "training_history.json").write_text(json.dumps(history_dict, indent=2), encoding="utf-8")

    cls_rep = classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"], digits=4)
    (reports_dir / "classification_report.txt").write_text(cls_rep, encoding="utf-8")

    with (reports_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["TN", "FP", "FN", "TP"])
        writer.writerow([int(tn), int(fp), int(fn), int(tp)])

    # Save false negatives & false positives for clinical error inspection
    with (reports_dir / "false_negatives.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "true_label", "predicted_label", "pneumonia_score"])
        for path, true_lbl, prob in zip(test_file_paths, y_true, y_prob):
            if true_lbl == 1 and prob < 0.5:
                writer.writerow([path, "PNEUMONIA", "NORMAL", f"{prob:.4f}"])
            elif true_lbl == 0 and prob >= 0.5:
                writer.writerow([path, "NORMAL", "PNEUMONIA", f"{prob:.4f}"])

    print("Evaluation and report generation completed.")


if __name__ == "__main__":
    main()