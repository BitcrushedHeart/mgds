import os
import shutil
import torch
import pytest
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule
from mgds.LoadingPipeline import LoadingPipeline
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.pipelineModules.SmartDiskCache import SmartDiskCache


class FakeSourceModule(PipelineModule, RandomAccessPipelineModule):
    """Minimal upstream module that provides file paths and fake tensor data.

    Simulates what CollectPaths + LoadImage + EncodeVAE would provide in production:
    - A file path (source_path_out_name)
    - Tensor data for split_names (e.g. 'latent_image')
    - Tensor data for aggregate_names (e.g. 'crop_resolution')
    """
    def __init__(self, file_dir: str, files: list[str], split_data: dict = None, aggregate_data: dict = None,
                 source_path_out_name: str = 'image_path'):
        super().__init__()
        self.file_dir = file_dir
        self.files = files
        self.split_data = split_data or {}
        self.aggregate_data = aggregate_data or {}
        self.source_path_out_name = source_path_out_name

    def length(self) -> int:
        return len(self.files)

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        outputs = [self.source_path_out_name]
        outputs.extend(self.split_data.keys())
        outputs.extend(self.aggregate_data.keys())
        return outputs

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        item = {self.source_path_out_name: os.path.join(self.file_dir, self.files[index])}
        for name, tensor_fn in self.split_data.items():
            item[name] = tensor_fn(variation, index)
        for name, tensor_fn in self.aggregate_data.items():
            item[name] = tensor_fn(variation, index)
        return item


def build_pipeline(modules, device='cpu', seed=42):
    """Wire up modules into a LoadingPipeline so SmartDiskCache can call _get_previous_item."""
    state = PipelineState(max_threads=1)
    pipeline = LoadingPipeline(
        device=torch.device(device),
        modules=modules,
        batch_size=1,
        seed=seed,
        state=state,
    )
    return pipeline


def create_test_files(directory, filenames, content_fn=None):
    """Create files in directory. content_fn(filename) -> bytes, defaults to filename as content."""
    os.makedirs(directory, exist_ok=True)
    for fname in filenames:
        path = os.path.join(directory, fname)
        if content_fn:
            data = content_fn(fname)
        else:
            data = fname.encode('utf-8')
        with open(path, 'wb') as f:
            f.write(data)


def make_latent_fn(shape=(1, 4, 8, 8)):
    """Returns a function that produces deterministic tensors from variation+index."""
    def fn(variation, index):
        torch.manual_seed(variation * 10000 + index)
        return torch.randn(shape)
    return fn


@pytest.fixture
def source_dir(tmp_path):
    d = tmp_path / "source"
    d.mkdir()
    return str(d)


@pytest.fixture
def cache_dir(tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    return str(d)
