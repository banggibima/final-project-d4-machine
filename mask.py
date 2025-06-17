import os
import json
from PIL import Image, ImageDraw

DATA_DIR = "new_data"
IMAGE_DIR = f"{DATA_DIR}/images"
MASK_DIR = f"{DATA_DIR}/masks"
TRAIN_SPLIT = "train"
VALID_SPLIT = "valid"
TEST_SPLIT = "test"

def load_coco_annotations(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def create_blank_mask(image_size):
    return Image.new('L', image_size, 0)

def draw_segmentation_on_mask(mask_img, segmentation, category_id):
    draw = ImageDraw.Draw(mask_img)
    for seg in segmentation:
        if len(seg) < 6:
            continue
        polygon = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
        draw.polygon(polygon, fill=int(category_id))

def draw_bbox_on_mask(mask_img, bbox, category_id):
    x, y, w, h = bbox
    draw = ImageDraw.Draw(mask_img)
    draw.rectangle([x, y, x + w, y + h], fill=int(category_id))

def generate_masks(coco_json_path, output_dir):
    coco = load_coco_annotations(coco_json_path)

    images_info = {img['id']: img for img in coco['images']}
    annotations = coco['annotations']

    os.makedirs(output_dir, exist_ok=True)

    for image_id, img_info in images_info.items():
        width, height = img_info['width'], img_info['height']
        mask = create_blank_mask((width, height))
        relevant_anns = [ann for ann in annotations if ann['image_id'] == image_id]

        for ann in relevant_anns:
            category_id = ann.get('category_id', 1)
            if 'segmentation' in ann and ann['segmentation']:
                draw_segmentation_on_mask(mask, ann['segmentation'], category_id)
            elif 'bbox' in ann and ann['bbox']:
                draw_bbox_on_mask(mask, ann['bbox'], category_id)

        img_filename = img_info['file_name']
        mask_filename = os.path.splitext(img_filename)[0] + '.png'
        mask.save(os.path.join(output_dir, mask_filename))

def main():
    splits = [TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT]
    for split in splits:
        coco_json_path = os.path.join(IMAGE_DIR, split, "_annotations.coco.json")
        mask_output_dir = os.path.join(MASK_DIR, split)
        generate_masks(coco_json_path, mask_output_dir)

if __name__ == '__main__':
    main()
