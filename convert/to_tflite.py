import torch
import torch.onnx
import argparse
import onnx
import shutil
import os
import tensorflow as tf
from pathlib import Path
from onnxsim import simplify
from onnx_tf.backend import prepare
from torchvision import models



def main(args):
    save_path = Path(args.output).parent
    save_path.mkdir(exist_ok=True)
    tflite_model_path = args.output
    onnx_model_path = args.output.replace('.tflite', '_temp.onnx')
    tf_model_path = Path(args.output.split('.')[0])
    if tf_model_path.exists(): shutil.rmtree(tf_model_path)
    
    model = models.__dict__[args.model]()
    model.eval()

    inputs = torch.randn(1, 3, *args.img_size)

    torch.onnx.export(
        model, 
        inputs, 
        onnx_model_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=13
    )

    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    onnx_model, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"

    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(str(tf_model_path))

    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path))
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_model = converter.convert()

    with open(str(tflite_model_path), 'wb') as f:
        f.write(tflite_model)

    shutil.rmtree(tf_model_path)
    os.remove(onnx_model_path)
    
    print(f"Finished converting and Saved model at {tflite_model_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--img-size', type=list, default=[224, 224])
    parser.add_argument('--output', type=str, default='output/resnet18.tflite')
    args = parser.parse_args()

    main(args)