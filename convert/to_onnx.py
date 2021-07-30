import torch
import torch.onnx
import argparse
import onnx
from pathlib import Path
from onnxsim import simplify
from torchvision import models


def main(args):
    save_path = Path(args.output).parent
    save_path.mkdir(exist_ok=True)
    
    model = models.__dict__[args.model]()
    model.eval()

    inputs = torch.randn(1, 3, *args.img_size)

    torch.onnx.export(
        model, 
        inputs, 
        args.output,
        input_names=['input'],
        output_names=['output'],
        opset_version=13
    )

    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)

    onnx_model, check = simplify(onnx_model)
    onnx.save(onnx_model, args.output)

    assert check, "Simplified ONNX model could not be validated"
    
    print(f"Finished converting and Saved model at {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--img-size', type=list, default=[224, 224])
    parser.add_argument('--output', type=str, default='output/resnet18.onnx')
    args = parser.parse_args()

    main(args)