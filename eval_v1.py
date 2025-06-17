# eval.py
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from keras import saving

# Path
DATASET_PATH = './data'
MODEL_DIR = './models_v1'
OUTPUT_DIR = './outputs_v1/eval'
os.makedirs(OUTPUT_DIR, exist_ok=True)

@saving.register_keras_serializable()
def dice_loss(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    dice = (2. * intersection + 1e-6) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-6
    )
    return 1 - dice

@saving.register_keras_serializable()
def bce_loss(y_true, y_pred):
    bce_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return bce_fn(y_true, y_pred)

@saving.register_keras_serializable()
def dice_bce_loss(y_true, y_pred):
    bce_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    bce = bce_fn(y_true, y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    dice = (2. * intersection + 1e-6) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-6
    )
    return bce + (1 - dice)

@saving.register_keras_serializable()
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    cross_entropy = -y_true * tf.math.log(y_pred)
    weight = alpha * tf.pow(1 - y_pred, gamma)
    loss = weight * cross_entropy
    return tf.reduce_mean(loss)

@saving.register_keras_serializable()
def tversky_loss(y_true, y_pred, alpha=0.5, beta=0.5):
    tp = tf.reduce_sum(y_true * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    tversky = (tp + 1e-6) / (tp + alpha * fn + beta * fp + 1e-6)
    return 1 - tversky

@saving.register_keras_serializable()
def dice_coef(y_true, y_pred):
    y_pred = tf.round(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + 1e-6) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-6
    )

@saving.register_keras_serializable()
def iou_metric(y_true, y_pred):
    y_pred = tf.round(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-6)

@saving.register_keras_serializable()
def precision_metric(y_true, y_pred):
    y_pred = tf.round(y_pred)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    return tp / (tp + fp + 1e-6)

@saving.register_keras_serializable()
def recall_metric(y_true, y_pred):
    y_pred = tf.round(y_pred)
    tp = tf.reduce_sum(y_true * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    return tp / (tp + fn + 1e-6)

@saving.register_keras_serializable()
def f1_score_metric(y_true, y_pred):
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall + 1e-6)

def load_image(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return img / 255.0

def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return np.expand_dims((mask > 0).astype(np.float32), axis=-1)

def load_dataset(image_dir, mask_dir):
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    images = [load_image(p) for p in image_paths]
    masks = [load_mask(p) for p in mask_paths]
    return np.array(images), np.array(masks), image_paths

def save_result(img, true_mask, pred_mask, path=None):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1); plt.imshow(img); plt.title("Image")
    plt.subplot(1, 3, 2); plt.imshow(true_mask.squeeze(), cmap='gray'); plt.title("Ground Truth")
    plt.subplot(1, 3, 3); plt.imshow(pred_mask.squeeze(), cmap='gray'); plt.title("Prediction")
    if path:
        plt.savefig(path, bbox_inches='tight', pad_inches=0.25)
        print(f"Result visualization saved to {path}")
    else:
        plt.show()

def main():
    model_path = os.path.join(MODEL_DIR, "unet_model_v001_best.keras")
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    image_dir = os.path.join(DATASET_PATH, "images/valid")
    mask_dir = os.path.join(DATASET_PATH, "masks/valid")
    X, y_true, image_paths = load_dataset(image_dir, mask_dir)

    print("Evaluating...")
    results = model.evaluate(X, y_true, verbose=1)
    for name, val in zip(model.metrics_names, results):
        print(f"{name}: {val:.4f}")

    print("Saving example predictions...")
    y_pred = model.predict(X)
    for i in range(min(5, len(X))):
        fname = os.path.basename(image_paths[i]).replace(".jpg", "_eval.png")
        save_path = os.path.join(OUTPUT_DIR, fname)
        pred_mask = (y_pred[i] > 0.5).astype(np.float32)
        save_result(X[i], y_true[i], pred_mask, save_path)

if __name__ == "__main__":
    main()
