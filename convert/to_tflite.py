import argparse
import tensorflow as tf
import numpy as np


def to_float32(converter):
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    return converter


def to_float16(converter):
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    return converter


def to_int8(converter, dataset_path):
    calibrate_data = np.load(dataset_path, allow_pickle=True)

    def representative_dataset_gen():
        for image in calibrate_data:
            image = image[None, ...]
            yield [image]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.int8
    converter.inference_input_type = tf.int8
    converter.representative_dataset = representative_dataset_gen
    return converter


def main(model_path, model_output_path, quant_type, dataset_path):    
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    
    if quant_type == 'float16':
        converter = to_float16(converter)
    elif quant_type == 'int8':
        assert dataset_path
        converter = to_int8(converter, dataset_path)
    else:
        converter = to_float32(converter)

    tflite_model = converter.convert()

    with open(model_output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Finished converting and Saved model.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='resnet18')
    parser.add_argument('--model-output-path', type=str, default='resnet18.tflite')
    parser.add_argument('--quant', type=str, default='float32')
    parser.add_argument('--dataset-path', type=str, default='')
    args = parser.parse_args()
    main(args.model_path, args.model_output_path, args.quant, args.dataset_path)