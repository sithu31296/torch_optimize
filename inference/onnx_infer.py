import torch
import argparse
import onnxruntime
import time
import numpy as np
from PIL import Image


class Inference:
    def __init__(self, model: str) -> None:
        # onnx model session
        self.session = onnxruntime.InferenceSession(model)
        
        # preprocess parameters
        model_inputs = self.session.get_inputs()[0]
        self.input_name = model_inputs.name
        self.img_size = model_inputs.shape[-2:]
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)

    def preprocess(self, image: Image.Image) -> np.ndarray:
        # resize
        image = image.resize(self.img_size)

        # to numpy array and to channel first
        image = np.array(image, dtype=np.float32).transpose(2, 0, 1)

        # scale to [0.0, 1.0]
        image /= 255

        # normalize
        image -= self.mean
        image /= self.std

        # add batch dimension 
        image = image[np.newaxis, ...]

        return image

    def postprocess(self, prob: np.ndarray) -> int:
        return np.argmax(prob)

    @torch.no_grad()
    def predict(self, img_path: str) -> int:
        image = Image.open(img_path).convert('RGB')
        image = self.preprocess(image)
        start = time.time()
        pred = self.session.run(None, {self.input_name: image})
        end = time.time()
        print(f"ONNX Model Inference Time: {(end-start)*1000:.2f}ms")

        cls_id = self.postprocess(pred)
        return cls_id

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='output/resnet18.onnx')
    parser.add_argument('--img-path', type=str, default='assests/dog.jpg')
    args = parser.parse_args()

    session = Inference(args.model)
    cls_id = session.predict(args.img_path)
    print(f"File: {args.img_path} >>>>> {cls_id}")
