# Optimize PyTorch Models

## Introduction

This project is for optimizing pytorch models for production. Optimization includes the following:

* Optimizing PyTorch models
* Converting to another frameworks (ONNX, TFLite, TensorRT, OpenVINO, NCNN, etc.)
* Optimizing converted models from another frameworks

## Installation

### Installing OpenVINO

**Download OpenVINO toolkit** from [here](https://software.intel.com/en-us/openvino-toolkit/choose-download).

On Linux:

```bash
$ tar -xvzf l_openvino_toolkit_p_<version>.tgz
$ cd l_openvino_toolkit_p_<version>
$ sudo ./install.sh
```

**[Optional] Install External Software Dependencies**

These include:
* Intel-optimized build of OpenCV library
* Inference Engine
* Model Optimizer Tools

On Linux:

```bash
$ cd /opt/intel/openvino_2021/install_dependencies
$ sudo -E ./install_openvino_dependencies.sh
```

**Set the Environment Variables**

* Open the `.bashrc` file.

```bash
$ gedit ~/.bashrc
```

* Add this line to the end of the file.

```bash
source /opt/intel/openvino_2021/bin/setupvars.sh
```

* Save and close the file.
* Open a new terminal and you will see `[setupvars.sh] OpenVINO environment initialized.`

**Configure the Model Optimizer**

* Go to the Model Optimizer pre-requisites directory.

```bash
$ cd /opt/intel/openvino_2021/deployment_tools/model_optimizer/install_prequisites
```

* Run the script for ONNX framework.

```bash
$ sudo ./install_prequisites_onnx.sh
```

**Uninstall OpenVINO**

Run the following command.

```bash
$ sudo /opt/intel/openvino_2021/openvino_toolkit_uninstaller/uninstall.sh -s
```

### Installing openvino2tensorflow

[openvino2tensorflow](https://github.com/PINTO0309/openvino2tensorflow) tool will be used to convert OpenVINO model to TensorFlow model. Install as follows:

```bash
$ pip install -U git+https://github.com/PINTO0309/openvino2tensorflow
```

## PyTorch to TFLite

### Step 1: Convert PyTorch to ONNX

```bash
$ python convert/to_onnx.py
```

### Step 2: Convert ONNX to OpenVINO

```bash
$ python <OpenVINO_INSTALL_DIR>/deployment_tools/model_optimizer/mo.py \
    --input_model <MODEL>.onnx \
    --output_dir <OpenVINO_MODEL_PATH> \
    --input_shape [B,C,H,W] \
    --data_type {FP16,FP32,half,float} \
```

### Step 3: Convert OpenVINO to TensorFlow

```bash
$ openvino2tensorflow \
    --model_path <OpenVINO_MODEL_PATH>/<MODEL>.xml \
    --model_output_path <TF_SAVED_MODEL_PATH> \
    --output_saved_model \
```

## Step 4: Convert TensorFlow to TFLite

```bash
$ python convert/to_tflite.py \
    --model-path <TF_SAVED_MODEL_PATH>
    --model-output-path <TFLITE_MODEL_PATH>
    --quant {'float32', 'float16', 'int8'}
```

> Notes: If you use int8 quantization, you need to add `--dataset-path <CALIBRATE_DATASET_PATH>` unlabelled data in numpy format.


## Benchmarks

CPU 

Methods | Inference Time (ms) | Model Size (ms) | Improvements (%)
--- | --- | --- | ---
original | - | - | -
orig+quantize | - | - | -
orig+prune | - | - | -
orig+quant+prune | - | - | -
orig2onnx | - | - | -
tflite | - | - | -
tflite+quantize | - | - | -

GPU

Methods | Inference Time (ms) | Model Size (ms) | Improvements (%)
--- | --- | --- | ---
original (FP32) | - | - | -
original (FP16) | - | - | -
tensorrt (FP32) | - | - | -
tensorrt (FP16) | - | - | -
tensorrt (int8) | - | - | -