# train.py
import os
import time
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, losses, saving, Model, Input, callbacks

tf.random.set_seed(42)
np.random.seed(42)

DATASET_PATH = './data'
MODEL_DIR = './models_v1'
PLOT_DIR = './outputs_v1/plots'
EPOCHS = 100
BATCH_SIZE = 2
DROPOUT_RATE = 0.3

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def get_next_version(base_dir, prefix, ext):
    existing = glob.glob(os.path.join(base_dir, f"{prefix}_v*{ext}"))
    versions = [int(os.path.basename(f).split("_v")[-1].split(ext)[0])
                for f in existing if f.split("_v")[-1].split(ext)[0].isdigit()]
    return max(versions, default=0) + 1

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

def plot_metrics(history, version):
    def save_plot(metric, label, color_train, color_val):
        plt.figure()
        plt.plot(history.history[metric], color=color_train, label=f"Training {label}")
        plt.plot(history.history[f'val_{metric}'], color=color_val, label=f"Validation {label}")
        plt.title(f'Training and Validation {label}')
        plt.xlabel('Epoch')
        plt.ylabel(label)
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, f"{metric}_unet_model_v{version:03d}.png")
        plt.savefig(path)
        print(f"{label} plot saved to {path}")
        plt.close()

    save_plot('accuracy', 'Accuracy', 'red', 'blue')
    save_plot('loss', 'Loss', 'red', 'blue')

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
def iou_metric(y_true, y_pred):
    y_pred = tf.round(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-6)

@saving.register_keras_serializable()
def dice_coef(y_true, y_pred):
    y_pred = tf.round(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + 1e-6) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-6
    )

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

def build_unet(input_size, dropout_rate=DROPOUT_RATE, loss_fn=dice_bce_loss):
    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.LeakyReLU(0.1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.LeakyReLU(0.1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.SpatialDropout2D(dropout_rate)(x)
        return x

    def attention_block(x, g, inter_channels):
        theta_x = layers.Conv2D(inter_channels, 1)(x)
        phi_g = layers.Conv2D(inter_channels, 1)(g)
        add = layers.Add()([theta_x, phi_g])
        act = layers.Activation('relu')(add)
        psi = layers.Conv2D(1, 1, activation='sigmoid')(act)
        return layers.Multiply()([x, psi])

    def up_block(x, skip, filters):
        x = layers.Conv2DTranspose(filters, 3, strides=2, padding='same')(x)
        attn = attention_block(skip, x, filters // 2)
        x = layers.Concatenate()([x, attn])
        x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
        return x

    inputs = Input(shape=input_size)
    c1 = conv_block(inputs, 32); p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 64);    p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 128);   p3 = layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, 256);   p4 = layers.MaxPooling2D()(c4)
    c5 = conv_block(p4, 512)

    u6 = up_block(c5, c4, 256)
    u7 = up_block(u6, c3, 128)
    u8 = up_block(u7, c2, 64)
    u9 = up_block(u8, c1, 32)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(u9)

    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss=loss_fn,
        metrics=['accuracy', iou_metric, dice_coef, precision_metric, recall_metric, f1_score_metric]
    )
    return model

def main():
    print("Loading dataset...")
    X_train, y_train = load_dataset(
        os.path.join(DATASET_PATH, 'images/train'),
        os.path.join(DATASET_PATH, 'masks/train')
    )
    X_val, y_val = load_dataset(
        os.path.join(DATASET_PATH, 'images/valid'),
        os.path.join(DATASET_PATH, 'masks/valid')
    )

    input_shape = X_train.shape[1:]
    print("Building model...")
    model = build_unet(input_size=input_shape)
    model.summary()

    version = get_next_version(MODEL_DIR, "unet_model", ".keras")
    model_path = os.path.join(MODEL_DIR, f"unet_model_v{version:03d}.keras")
    checkpoint_path = os.path.join(MODEL_DIR, f"unet_model_v{version:03d}_best.keras")

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)

    print("Training model...")
    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, checkpoint]
    )
    print(f"Training finished in {time.time() - start:.2f} seconds")

    print(f"Saving model to {model_path}")
    model.save(model_path)

    plot_metrics(history, version)

if __name__ == '__main__':
    main()
