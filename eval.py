import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import pytesseract
from keras import models, losses, saving

DATASET_PATH = './data'
MODEL_PATH = './models/unet_model_epoch100_best.keras'
RESULT_DIR = './outputs/results'
INVERT_DIR = './outputs/inverts'
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(INVERT_DIR, exist_ok=True)

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
    return np.array(images), np.array(masks)

def postprocess_mask(pred_mask, threshold=0.5):
    mask = (pred_mask.squeeze() > threshold).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def save_result(img, true_mask, pred_mask, path):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(true_mask.squeeze(), cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask.squeeze(), cmap='gray')
    plt.title("Prediction")
    plt.axis('off')

    plt.savefig(path, bbox_inches='tight', pad_inches=0.25)
    print(f"Result saved to {path}")
    plt.close()

def run_ocr(image_rgb, mask_binary):
    # Apply mask to image (bitwise AND)
    mask_bool = (mask_binary > 0).astype(np.uint8)
    masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_bool)

    # Convert to grayscale for OCR
    gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)

    # Optional preprocessing for OCR
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Run OCR
    text = pytesseract.image_to_string(gray, config='--psm 7')
    return text.strip()

@saving.register_keras_serializable()
def dice_bce_loss(y_true, y_pred):
    bce = losses.binary_crossentropy(y_true, y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    dice = (2. * intersection + 1e-6) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-6)
    return 1 - dice + bce

@saving.register_keras_serializable()
def iou_metric(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-6)

def main():
    print("Loading test dataset...")
    X_test, y_test = load_dataset(
        os.path.join(DATASET_PATH, 'images/test'),
        os.path.join(DATASET_PATH, 'masks/test')
    )

    print(f"Loading model from {MODEL_PATH}...")
    model = models.load_model(MODEL_PATH, custom_objects={
        'dice_bce_loss': dice_bce_loss,
        'iou_metric': iou_metric
    })

    print("Evaluating model...")
    loss, acc, iou = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f} | Accuracy: {acc:.4f} | IOU: {iou:.4f}")

    print("Saving predictions, inverted masks, and OCR results...")
    for i in range(min(5, len(X_test))):
        pred = model.predict(np.expand_dims(X_test[i], axis=0))[0]
        clean_mask = postprocess_mask(pred)

        # Save result visualization
        save_path = os.path.join(RESULT_DIR, f'prediction_{i}.png')
        save_result(X_test[i], y_test[i], clean_mask / 255.0, save_path)

        # Save inverted mask
        inverted_mask = 255 - clean_mask
        invert_path = os.path.join(INVERT_DIR, f'inverted_{i}.png')
        cv2.imwrite(invert_path, inverted_mask)
        print(f"Inverted mask saved to {invert_path}")

        # OCR
        img_uint8 = (X_test[i] * 255).astype(np.uint8)
        ocr_text = run_ocr(img_uint8, clean_mask)
        print(f"OCR Result [{i}]:", repr(ocr_text))

if __name__ == '__main__':
    main()
