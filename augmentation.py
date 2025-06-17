import os
import cv2
import albumentations as A
import numpy as np
from tqdm import tqdm

# Subset yang ingin di-augment
SUBSETS = ['train', 'valid', 'test']

# Output direktori
OUTPUT_BASE = 'new_data_v1'

# Semua augmentasi spesifik
augmentations = [
    ("rot_minus10", A.Rotate(limit=(-10, -10), p=1.0)), # Rotate -10 degrees
    ("rot_plus10", A.Rotate(limit=(10, 10), p=1.0)), # Rotate +10 degrees
    ("rot_minus15", A.Rotate(limit=(-15, -15), p=1.0)), # Rotate -15 degrees
    ("rot_plus15", A.Rotate(limit=(15, 15), p=1.0)), # Rotate +15 degrees

    ("shear_11_11", A.Affine(shear={"x": 11, "y": 11}, p=1.0)), # Shear 11 degrees in both x and y
    ("shear_11_-11", A.Affine(shear={"x": 11, "y": -11}, p=1.0)), # Shear 11 degrees in x, -11 in y
    ("shear_-11_11", A.Affine(shear={"x": -11, "y": 11}, p=1.0)), # Shear -11 degrees in x, 11 in y
    ("shear_-11_-11", A.Affine(shear={"x": -11, "y": -11}, p=1.0)), # Shear -11 degrees in both x and y
    ("shear_12_12", A.Affine(shear={"x": 12, "y": 12}, p=1.0)), # Shear 12 degrees in both x and y
    ("shear_12_-12", A.Affine(shear={"x": 12, "y": -12}, p=1.0)), # Shear 12 degrees in x, -12 in y
    ("shear_-12_12", A.Affine(shear={"x": -12, "y": 12}, p=1.0)), # Shear -12 degrees in x, 12 in y
    ("shear_-12_-12", A.Affine(shear={"x": -12, "y": -12}, p=1.0)), # Shear -12 degrees in both x and y
    ("shear_13_13", A.Affine(shear={"x": 13, "y": 13}, p=1.0)), # Shear 13 degrees in both x and y
    ("shear_13_-13", A.Affine(shear={"x": 13, "y": -13}, p=1.0)), # Shear 13 degrees in x, -13 in y
    ("shear_-13_13", A.Affine(shear={"x": -13, "y": 13}, p=1.0)), # Shear -13 degrees in x, 13 in y
    ("shear_-13_-13", A.Affine(shear={"x": -13, "y": -13}, p=1.0)), # Shear -13 degrees in both x and y
    ("shear_14_14", A.Affine(shear={"x": 14, "y": 14}, p=1.0)), # Shear 14 degrees in both x and y
    ("shear_14_-14", A.Affine(shear={"x": 14, "y": -14}, p=1.0)), # Shear 14 degrees in x, -14 in y
    ("shear_-14_14", A.Affine(shear={"x": -14, "y": 14}, p=1.0)), # Shear -14 degrees in x, 14 in y
    ("shear_-14_-14", A.Affine(shear={"x": -14, "y": -14}, p=1.0)), # Shear -14 degrees in both x and y
    ("shear_15_15", A.Affine(shear={"x": 15, "y": 15}, p=1.0)), # Shear 15 degrees in both x and y
    ("shear_15_-15", A.Affine(shear={"x": 15, "y": -15}, p=1.0)), # Shear 15 degrees in x, -15 in y
    ("shear_-15_15", A.Affine(shear={"x": -15, "y": 15}, p=1.0)), # Shear -15 degrees in x, 15 in y
    ("shear_-15_-15", A.Affine(shear={"x": -15, "y": -15}, p=1.0)), # Shear -15 degrees in both x and y

    ("bright_-10", A.RandomBrightnessContrast(brightness_limit=(-0.1, -0.1), contrast_limit=0.0, p=1.0)), # Brightness -10%
    ("bright_+10", A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1), contrast_limit=0.0, p=1.0)), # Brightness +10%
    ("bright_-20", A.RandomBrightnessContrast(brightness_limit=(-0.2, -0.2), contrast_limit=0.0, p=1.0)), # Brightness -20%
    ("bright_+20", A.RandomBrightnessContrast(brightness_limit=(0.2, 0.2), contrast_limit=0.0, p=1.0)), # Brightness +20%

    ("gamma_-10", A.RandomGamma(gamma_limit=(90, 90), p=1.0)),   # Exposure -10%
    ("gamma_+10", A.RandomGamma(gamma_limit=(110, 110), p=1.0)), # Exposure +10%
    ("gamma_-15", A.RandomGamma(gamma_limit=(85, 85), p=1.0)),   # Exposure -15%
    ("gamma_+15", A.RandomGamma(gamma_limit=(115, 115), p=1.0)), # Exposure +15%

    ("crop_center_15", A.Compose([
        A.CenterCrop(height=int(0.85 * 512), width=int(0.85 * 512), p=1.0),
        A.Resize(height=640, width=640, p=1.0)
    ])),

    ("crop_center_30", A.Compose([
        A.CenterCrop(height=int(0.70 * 512), width=int(0.70 * 512), p=1.0),
        A.Resize(height=640, width=640, p=1.0)
    ])), # Crop center 30% from 100% image

    ("gray_15", A.ToGray(p=1.0)), # Convert to grayscale
    ("blur_2_5px", A.GaussianBlur(blur_limit=(5, 5), p=1.0)),  # Gaussian blur approx. radius 2.5px
]

# Fungsi augmentasi untuk satu subset
def augment_subset(subset):
    print(f"Memproses subset: {subset}")

    input_img_dir = f"new_data/images/{subset}"
    input_mask_dir = f"new_data/masks/{subset}"
    output_img_dir = f"{OUTPUT_BASE}/images/{subset}"
    output_mask_dir = f"{OUTPUT_BASE}/masks/{subset}"

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_img_dir) if f.endswith('.jpg')]

    for img_name in tqdm(image_files):
        img_path = os.path.join(input_img_dir, img_name)
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(input_mask_dir, mask_name)

        # Load image dan mask
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Simpan original
        cv2.imwrite(os.path.join(output_img_dir, img_name), image)
        cv2.imwrite(os.path.join(output_mask_dir, mask_name), mask)

        # Lakukan semua augmentasi yang ditentukan
        for tag, aug in augmentations:
            composed = A.Compose([aug], additional_targets={"mask": "mask"})
            augmented = composed(image=image, mask=mask)
            aug_img = augmented['image']
            aug_mask = augmented['mask']

            # Buat nama baru
            aug_img_name = img_name.replace('.jpg', f'_{tag}.jpg')
            aug_mask_name = mask_name.replace('.png', f'_{tag}.png')

            cv2.imwrite(os.path.join(output_img_dir, aug_img_name), aug_img)
            cv2.imwrite(os.path.join(output_mask_dir, aug_mask_name), aug_mask)

# Proses semua subset
for subset in SUBSETS:
    augment_subset(subset)

print("Augmentasi selesai. Hasil disimpan di folder data_v1/")