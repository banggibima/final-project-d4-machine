# train.py
import os
import time
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, losses, saving, Model, Input, callbacks

DATASET_PATH = './data'
MODEL_DIR = './models'
PLOT_DIR = './outputs/plots'
EPOCHS = 100
BATCH_SIZE = 2
DROPOUT_RATE = 0.3

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def get_next_version(base_dir, prefix, ext):
    existing_files = glob.glob(os.path.join(base_dir, f"{prefix}_v*{ext}"))
    versions = []
    for f in existing_files:
        basename = os.path.basename(f)
        try:
            v_str = basename.split("_v")[-1].split(ext)[0]
            versions.append(int(v_str))
        except ValueError:
            continue
    next_version = max(versions, default=0) + 1
    return next_version

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

def plot_history(hist, base_path):
    plt.figure()
    plt.plot(hist.history['accuracy'], 'r', label='Train Acc')
    plt.plot(hist.history['val_accuracy'], 'b', label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    acc_path = os.path.join(base_path, 'accuracy_unet_model_epoch100.png')
    plt.savefig(acc_path)
    print(f"Accuracy plot saved to {acc_path}")
    plt.close()

    plt.figure()
    plt.plot(hist.history['loss'], 'r', label='Train Loss')
    plt.plot(hist.history['val_loss'], 'b', label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    loss_path = os.path.join(base_path, 'loss_unet_model_epoch100.png')
    plt.savefig(loss_path)
    print(f"Loss plot saved to {loss_path}")
    plt.close()

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

def build_unet(input_size, dropout_rate=DROPOUT_RATE):
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

    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, 256)
    p4 = layers.MaxPooling2D()(c4)
    c5 = conv_block(p4, 512)
    u6 = up_block(c5, c4, 256)
    u7 = up_block(u6, c3, 128)
    u8 = up_block(u7, c2, 64)
    u9 = up_block(u8, c1, 32)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(u9)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss=dice_bce_loss, metrics=['accuracy', iou_metric])
    return model

def main():
    print("Loading dataset...")
    X_train, y_train = load_dataset(
        os.path.join(DATASET_PATH, 'images/train'),
        os.path.join(DATASET_PATH, 'masks/train')
    )
    X_valid, y_valid = load_dataset(
        os.path.join(DATASET_PATH, 'images/valid'),
        os.path.join(DATASET_PATH, 'masks/valid')
    )

    input_shape = X_train.shape[1:]
    print("Building model...")
    model = build_unet(input_size=input_shape)
    model.summary()

    version = get_next_version(MODEL_DIR, "unet_model", ".keras")
    model_name = f"unet_model_v{version:03d}.keras"
    model_path = os.path.join(MODEL_DIR, model_name)
    checkpoint_name = f"unet_model_v{version:03d}_best.keras"
    checkpoint_path = os.path.join(MODEL_DIR, checkpoint_name)
    plot_acc_path = os.path.join(PLOT_DIR, f"accuracy_unet_model_v{version:03d}.png")
    plot_loss_path = os.path.join(PLOT_DIR, f"loss_unet_model_v{version:03d}.png")

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)

    print("Training model...")
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, model_checkpoint]
    )
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

    print(f"Saving final model to {model_path}")
    model.save(model_path)

    plt.figure()
    plt.plot(history.history['accuracy'], 'r', label='Train Acc')
    plt.plot(history.history['val_accuracy'], 'b', label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(plot_acc_path)
    print(f"Accuracy plot saved to {plot_acc_path}")
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'], 'r', label='Train Loss')
    plt.plot(history.history['val_loss'], 'b', label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(plot_loss_path)
    print(f"Loss plot saved to {plot_loss_path}")
    plt.close()

if __name__ == '__main__':
    main()
