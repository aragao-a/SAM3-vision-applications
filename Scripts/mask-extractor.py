import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "campo-casa.jpeg"
PROMPT_TEXT = "house"
OUTPUT_FOLDER = BASE_DIR / "results"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def run_extractor():

    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    image = Image.open(IMAGE_PATH).convert("RGB")
    img_np = np.array(image)
    inference_state = processor.set_image(image)

    output = processor.set_text_prompt(state=inference_state, prompt=PROMPT_TEXT)

    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]

    plt.figure(figsize=(12, 8))
    plt.imshow(image)

    for i, (mask_tensor, box_tensor) in enumerate(zip(masks, boxes)):
        mask = mask_tensor.cpu().numpy().squeeze()
        box = box_tensor.cpu().numpy().astype(int)
        score = scores[i].item()

        mask_visual = np.zeros((mask.shape[0], mask.shape[1], 4))
        mask_visual[mask > 0] = [0, 1, 0, 0.4]
        plt.imshow(mask_visual)

        alpha = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        alpha[mask > 0] = 255

        rgba_img = np.dstack((img_np, alpha))
        crop = rgba_img[box[1]:box[3], box[0]:box[2]]

        cutout_filename = OUTPUT_FOLDER / f"recorte_{PROMPT_TEXT}_{timestamp}_{i}.png"
        Image.fromarray(crop, "RGBA").save(cutout_filename)

    vis_filename = OUTPUT_FOLDER / f"visualizacao_{PROMPT_TEXT}_{timestamp}.png"
    plt.title(f"SAM 3: {PROMPT_TEXT} | {timestamp}")
    plt.axis('off')
    plt.savefig(vis_filename)
    plt.close() 
    
    print("feito")

if __name__ == "__main__":
    run_extractor()