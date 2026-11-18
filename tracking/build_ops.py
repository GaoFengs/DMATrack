import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from torch.utils.cpp_extension import load
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU and the CUDA 11.1 development toolkit are required')
    source = ROOT / 'external/prroi_pool/src'
    build = ROOT / '.build/prroi'
    build.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MAX_JOBS', '2')
    operation = load('_prroi_pooling', [str(source/'prroi_pooling_gpu.c'),
                     str(source/'prroi_pooling_gpu_impl.cu')],
                     build_directory=str(build), verbose=True)
    from external.prroi_pool import functional
    functional._prroi_pooling = operation
    features = torch.arange(64, device='cuda', dtype=torch.float32).view(1, 1, 8, 8).requires_grad_()
    rois = torch.tensor([[0., 1., 1., 6., 6.]], device='cuda')
    pooled = functional.prroi_pool2d(features, rois, 4, 4, 1.)
    pooled.sum().backward()
    assert torch.isfinite(pooled).all() and torch.isfinite(features.grad).all()
    print('PreciseRoIPooling forward/backward passed:', operation.__file__)


if __name__ == '__main__':
    main()
