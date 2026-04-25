import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from u2net import U2NET


class Segmenter:
    def __init__(self, model_path="models/u2net.pth"):
        self.net = U2NET(3, 1)
        self.net.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=False)
        )
        self.net.eval()

        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor()
        ])

    def predict(self, image_path):
        # Load image
        img = Image.open(image_path).convert("RGB")
        orig = np.array(img)

        # Preprocess
        inp = self.transform(img).unsqueeze(0)

        # Inference
        with torch.no_grad():
            d1, *_ = self.net(inp)

        # Get prediction
        pred = d1[:, 0, :, :]
        pred = pred.squeeze().cpu().numpy()

        # 🔥 SAFE NORMALIZATION (important fix)
        if pred.max() > pred.min():
            pred = (pred - pred.min()) / (pred.max() - pred.min())

        # Resize to original size
        mask = cv2.resize(pred, (orig.shape[1], orig.shape[0]))

        # Convert to binary mask
        mask = (mask > 0.5).astype(np.uint8) * 255

        return mask


if __name__ == "__main__":
    seg = Segmenter("models/u2net.pth")

    mask = seg.predict("test_images/image1.jpg")

    cv2.imwrite("output/body_mask.png", mask)

    print("Mask saved at output/body_mask.png")