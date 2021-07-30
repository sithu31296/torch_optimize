# Optimize PyTorch Models

* [Introduction](#introduction)
* [Installation](#installation)
* [Features](#features)
* [Benchmarks](#benchmarks)

## Introduction

This project is for optimizing pytorch models for production. Optimization includes the following:

* Optimizing PyTorch models
* Converting to another frameworks (ONNX, TFLite, TensorRT, OpenVINO, NCNN, etc.)
* Optimizing converted models from another frameworks

## Installation

Coming Soon...


## Features

* Conversion
    * [PyTorch to ONNX](./convert/to_onnx.py)
    * [PyTorch to TFLite](./convert/to_tflite.py)
    * [PyTorch to TensorRT]() (Coming Soon)

* Inference
    * [PyTorch](./inference/pt_infer.py)
    * [ONNX](./inference/onnx_infer.py)
    * [TFLite](./inference/tflite_infer.py)
    * [TensorRT]() (Coming Soon)

* Quantization
    * [PyTorch](./optimize/quantize.py)
    * [TFLite]() (Coming Soon)
    * [TensorRT]() (Coming Soon)

* Pruning
    * [PyTorch](./optimize/prune.py)

* Model Inspection
    * [Benchmark](./inspect/benchmark_model.py)
    * [Profiler](./inspect/profile_model.py)


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