Repository README — plantvillage-potato-annotation

Short description: Sample repository showing annotation work for potato Early Blight segmentation (examples, annotation formats, and automation scripts).

Overview

This repository demonstrates a full annotation workflow for agricultural disease segmentation: manual annotation in CVAT (Cloud & Local), semi-automated mask generation with ViT → SAM on Google Colab, conversion of masks to polygons, and validated upload to CVAT Cloud. The repo contains small, anonymized samples suitable for sharing.

Dataset (samples)

data_samples/ — small set of raw images (cropped / anonymized) used for examples (recommended: 10–30 images).

annotations/ — example annotation files (COCO JSON sample, Pascal VOC XML sample).

overlays/ — image + mask overlays (PNG) for quick visual proof of work.

Annotation formats

COCO (JSON) — sample: annotations/sample_coco.json

Pascal VOC (XML) — sample: annotations/sample_voc.xml

KITTI / nuScenes examples if available in annotations/kitti_nuscenes/

Repository structure
plantvillage-potato-annotation/
├─ data_samples/
│  ├─ sample_001.jpg
│  ├─ sample_002.jpg
├─ annotations/
│  ├─ sample_coco.json
│  ├─ sample_voc.xml
├─ overlays/
│  ├─ sample_001_overlay.png
├─ scripts/
│  ├─ sam_segmentation.ipynb
│  ├─ upload_to_cvat.py
│  ├─ visualize_overlay.py
├─ README.md
├─ LICENSE
How to reproduce (local)

Clone the repository:

git clone https://github.com/USERNAME/plantvillage-potato-annotation.git
cd plantvillage-potato-annotation

Create a virtual environment and install requirements (if you add requirements.txt):

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Quick visualization script (example)

visualize_overlay.py demonstrates how to overlay masks on images (example code):

from PIL import Image
import numpy as np


def overlay_mask(image_path, mask_path, alpha=0.5):
    img = Image.open(image_path).convert('RGBA')
    mask = Image.open(mask_path).convert('RGBA')
    mask_arr = np.array(mask)
    # Colorize mask (red) keeping transparency from mask alpha
    color = np.zeros_like(mask_arr)
    color[...,0] = 255  # red channel
    color[...,3] = (mask_arr[...,0] > 0).astype('uint8') * int(255*alpha)
    overlay = Image.fromarray(color, 'RGBA')
    result = Image.alpha_composite(img, overlay)
    result.save(image_path.replace('.jpg','_overlay.png'))


if __name__ == '__main__':
    overlay_mask('data_samples/sample_001.jpg', 'annotations/sample_mask_001.png')
Automation notebooks

sam_segmentation.ipynb — Colab notebook that runs ViT attention extraction + SAM mask generation and saves mask PNGs.

upload_to_cvat.py — script that converts masks to polygons and uploads to CVAT Cloud using the REST API / SDK (check and configure CVAT_URL, USERNAME, TOKEN).

License

Choose a license (recommended: MIT). If you want, add a short LICENSE file.

Contact

Abu Arab — add your email and GitHub profile link in the resume and here.

Notes:

Only include anonymized or small sample images. Do not upload sensitive or proprietary datasets.

Replace USERNAME and placeholders with your actual GitHub username and data paths.
