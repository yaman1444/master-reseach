import cv2
import os

images_to_fix = {
    'results/prediction_malignant (46).png': 'Malin'
}

for img_path, label in images_to_fix.items():
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        # The matplotlib title is typically at the top. Let's cover the top 45 pixels with white.
        # But wait, we don't know the exact height. Let's cover top 60 pixels just to be safe.
        # The original image might be 1600x600 (16x6 inches with 100 dpi).
        # Let's draw a white rectangle over the top part of the center panel.
        # Actually, if it's a 3-panel matplotlib figure, the title is above the middle panel.
        # Just cover the top 60 pixels across the entire width.
        h, w, _ = img.shape
        cv2.rectangle(img, (0, 0), (w, 60), (255, 255, 255), -1)
        
        # Now write the new title in the middle
        text = f"Grad-CAM: {label.upper()}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        text_x = (w - text_size[0]) // 2
        text_y = 40
        
        cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        
        out_path = f"results/gradcam_{label.lower()}_final.png"
        cv2.imwrite(out_path, img)
        print(f"Fixed {img_path} -> {out_path}")
    else:
        print(f"File not found: {img_path}")
