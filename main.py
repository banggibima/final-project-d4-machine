from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from keras import models, losses, saving
from io import BytesIO
from PIL import Image
import uvicorn
import numpy as np
import cv2
import re
import os
import tensorflow as tf
import pytesseract
import easyocr

try:
    easyocr_reader = easyocr.Reader(['en'], gpu=False)
except ImportError:
    easyocr_reader = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

MODEL_PATH = './models/unet_model_epoch100_best.keras'
OUTPUT_DIR = './outputs/predict'
INPUT_SHAPE = (640, 640)

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

model = models.load_model(MODEL_PATH, custom_objects={
    'dice_bce_loss': dice_bce_loss,
    'iou_metric': iou_metric
})

def postprocess_mask(pred_mask, threshold=0.5):
    mask = (pred_mask.squeeze() > threshold).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def run_tesseract(image_rgb, mask_binary):
    mask_bool = (mask_binary > 0).astype(np.uint8)
    masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_bool)
    gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(gray, config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    cleaned = re.sub(r'[^A-Za-z0-9]', '', text)
    return cleaned.strip()

def run_easyocr(image_rgb, mask_binary):
    if easyocr_reader is None:
        return "EasyOCR not installed"
    mask_bool = (mask_binary > 0).astype(np.uint8)
    masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_bool)
    result = easyocr_reader.readtext(masked_image, detail=1)
    text = ''.join([re.sub(r'[^A-Za-z0-9]', '', item[1]) for item in result])
    return text.strip()

def save_mask_outputs(mask, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = os.path.join(output_dir, f"pred_mask_{timestamp}.png")
    inv_path = os.path.join(output_dir, f"invert_mask_{timestamp}.png")
    cv2.imwrite(pred_path, mask)
    cv2.imwrite(inv_path, 255 - mask)
    return pred_path, inv_path

@app.post("/predict/")
async def predict(file: UploadFile = File(...), ocr_engine: str = Query("tesseract", enum=["tesseract", "easyocr"])):
    contents = await file.read()
    pil_img = Image.open(BytesIO(contents)).convert("RGB")
    orig_size = pil_img.size

    img = np.array(pil_img)
    img_resized = cv2.resize(img, INPUT_SHAPE) / 255.0
    input_tensor = np.expand_dims(img_resized, axis=0)

    pred = model.predict(input_tensor)[0]
    pred_mask = postprocess_mask(pred)

    mask_original_size = cv2.resize(pred_mask, orig_size[::-1], interpolation=cv2.INTER_NEAREST)
    pred_path, inv_path = save_mask_outputs(mask_original_size)

    if ocr_engine == "easyocr":
        ocr_text = run_easyocr(img, mask_original_size)
    else:
        ocr_text = run_tesseract(img, mask_original_size)

    return JSONResponse({
        "ocr_result": ocr_text,
        "ocr_engine": ocr_engine,
        "mask_path": pred_path,
        "invert_mask_path": inv_path
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
