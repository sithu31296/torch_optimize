"""PyTorch Profiler
    -   To measure the time and memory consumption of the model's operators.
    -   Is useful when user needs to determine the most expensive operators in the model.

"""
import argparse
import torch
from pathlib import Path
from torchvision import models
from torch.profiler import profile, record_function, ProfilerActivity

import sys
sys.path.insert(0, '.')
from utils.utils import setup_cudnn



def profile_model(model, inputs, device='cpu', profile_memory=True, profile_shape=False, sort_by='time_total', save_json=''):
    assert sort_by in ['time_total', 'memory_usage']
    assert device in ['cpu', 'cuda']

    if device == 'cpu':
        activities = [ProfilerActivity.CPU]
    else:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    # note: the first use of CUDA profiling may bring an extra overhead
    with profile(activities=activities, record_shapes=True, profile_memory=profile_memory) as prof:
        with record_function("model_inference"):
            model(inputs)

    print(prof.key_averages(group_by_input_shape=profile_shape).table(sort_by=f"{device}_{sort_by}", row_limit=10))

    if save_json.endswith('.json'):
        prof.export_chrome_trace(save_json)


def main(args):
    setup_cudnn()

    if args.save:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(exist_ok=True)
        save_json = save_dir / f"{args.model}_trace.json"
    else:
        save_json = ''

    model = models.__dict__[args.model]()
    inputs = torch.randn(1, 3, *args.img_size)

    profile_model(
        model, 
        inputs, 
        args.device, 
        args.memory, 
        args.shape, 
        args.sort, 
        str(save_json)
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--img-size', type=list, default=[224, 224])
    parser.add_argument('--memory', type=bool, default=True)
    parser.add_argument('--shape', type=bool, default=False)
    parser.add_argument('--sort', type=str, default='time_total')
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--save-dir', type=str, default='output')
    args = parser.parse_args()

    main(args)