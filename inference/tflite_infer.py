import torch
import argparse
import time
import tensorflow as tf
import numpy as np
from PIL import Image


class Inference:
    def __init__(self, model: str) -> None:
        # tflite interpreter
        self.interpreter = tf.lite.Interpreter(model)
        self.interpreter.allocate_tensors()
        
        # preprocess and postprocess parameters
        input_details = self.interpreter.get_input_details()[0]
        output_details = self.interpreter.get_output_details()[0]
        self.input_index = input_details['index']
        self.output_index = output_details['index']
        self.img_size = input_details['shape'][-2:]
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
        self.interpreter.set_tensor(self.input_index, image)
        self.interpreter.invoke()
        pred = self.interpreter.get_tensor(self.output_index)
        end = time.time()
        print(f"TFLite Model Inference Time: {(end-start)*1000:.2f}ms")

        cls_id = self.postprocess(pred)
        return cls_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='output/resnet18.tflite')
    parser.add_argument('--img-path', type=str, default='assests/dog.jpg')
    args = parser.parse_args()

    session = Inference(args.model)
    cls_id = session.predict(args.img_path)
    print(f"File: {args.img_path} >>>>> {cls_id}")
