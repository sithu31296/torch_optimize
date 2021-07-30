import torch
import argparse
from torch import Tensor
from torchvision import io
from torchvision import transforms as T
from torchvision import models
import sys
sys.path.insert(0, '.')
from utils.utils import time_sync


class Inference:
    def __init__(self, device: str, model: str, img_size: list) -> None:
        self.device = torch.device(device)
        self.model = models.__dict__[model]()
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize(img_size),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def preprocess(self, image: Tensor) -> Tensor:
        # scale to [0.0, 1.0]
        image = image.float()
        image /= 255

        # resize and normalize
        image = self.transform(image)

        # add batch dimension and send to device
        image = image.unsqueeze(0).to(self.device)

        return image

    def postprocess(self, prob: Tensor) -> int:
        return torch.argmax(prob)

    @torch.no_grad()
    def predict(self, img_path: str) -> int:
        image = io.read_image(img_path)
        image = self.preprocess(image)
        start = time_sync()
        pred = self.model(image)
        end = time_sync()
        print(f"PyTorch Model Inference Time: {(end-start)*1000:.2f}ms")

        cls_id = self.postprocess(pred)
        return cls_id

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--img-size', type=list, default=[224, 224])
    parser.add_argument('--img-path', type=str, default='assests/dog.jpg')
    args = parser.parse_args()

    session = Inference(args.device, args.model, args.img_size)
    cls_id = session.predict(args.img_path)
    print(f"File: {args.img_path} >>>>> {cls_id}")
