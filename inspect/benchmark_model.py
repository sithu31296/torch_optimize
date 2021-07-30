import torch
import argparse
import yaml
from torch import nn, Tensor
from torch.utils import benchmark
from torchvision import models
import sys
sys.path.insert(0, '.')
from utils.utils import setup_cudnn


def benchmark_model(model: nn.Module, inputs: Tensor, times: int = 100, num_threads: int = None, wall_time: bool = False):
    if num_threads is None:
        num_threads = torch.get_num_threads()
    
    print(f'Benchmarking with {num_threads} threads for {times} times.')
    timer = benchmark.Timer(
        stmt=f"{model}(x)",             # computation which will be run in a loop and times
        setup= f"x = {inputs}",         # setup will be run before calling the measurement loop and is used to populate any state which is need by 'stmt'
        num_threads=num_threads
    )
    if wall_time:
        return timer.blocked_autorange(min_run_time=0.2)
    return timer.timeit(times)


def main(args):
    setup_cudnn()

    model = models.__dict__[args.model]()
    inputs = torch.randn(1, 3, *args.img_size)

    time = benchmark_model(model, inputs)
    print(time)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--img-size', type=list, default=[224, 224])
    args = parser.parse_args()

    main(args)