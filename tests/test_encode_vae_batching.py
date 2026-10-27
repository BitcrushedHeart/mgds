"""
Batch-vs-single equivalence tests for EncodeVAE's batch collector.

With the collector enabled and requests of mixed image shapes arriving
concurrently, every image's latent distribution must match the legacy bs=1
encode, batches must never mix shapes or exceed the pixel budget, and each
returned distribution must own its storage (torch.save writes a view's whole
storage, so a view into the batch would bloat every cached .pt).
"""

import threading
from concurrent.futures import ThreadPoolExecutor

from mgds.MGDS import MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModules.EncodeVAE import EncodeVAE
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch

import pytest

SHAPES = [(3, 32, 32), (3, 16, 32), (3, 32, 16)]
NUM_ITEMS = 12


class ImageSource(PipelineModule, RandomAccessPipelineModule):
    def __init__(self, images: list[torch.Tensor]):
        super().__init__()
        self.images = images

    def length(self) -> int:
        return len(self.images)

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return ["image"]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {"image": self.images[index]}


@pytest.fixture(scope="module")
def images():
    g = torch.Generator()
    g.manual_seed(1)
    return [torch.rand(SHAPES[i % len(SHAPES)], generator=g) * 2 - 1 for i in range(NUM_ITEMS)]


@pytest.fixture(scope="module")
def tiny_vae():
    from diffusers import AutoencoderKL

    torch.manual_seed(0)
    return AutoencoderKL(
        block_out_channels=(8, 16),
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        latent_channels=4,
        norm_num_groups=4,
        sample_size=32,
    ).eval()


class RecordingVAE(torch.nn.Module):
    """Wraps a VAE to record the batch shape of every encode call."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self.batch_shapes: list[tuple] = []
        self._lock = threading.Lock()

    def encode(self, x):
        with self._lock:
            self.batch_shapes.append(tuple(x.shape))
        return self.vae.encode(x)


def _build_pipeline(images, encode_module):
    output = OutputPipelineModule(names=["latent"])
    MGDS(
        device=torch.device("cpu"),
        concepts=[{"name": "C", "path": "dummy"}],
        settings={},
        definition=[[ImageSource(images)], [encode_module], [output]],
        batch_size=1,
        state=PipelineState(),
        seed=7,
    )
    return encode_module


def _concurrent_results(module, num_items):
    def fetch(i):
        with torch.no_grad():
            return i, module.get_item(0, i)["latent"]

    with ThreadPoolExecutor(max_workers=num_items) as pool:
        return dict(pool.map(fetch, range(num_items)))


class TestEncodeVAEBatching:
    def test_concurrent_batched_matches_single(self, tiny_vae, images):
        baseline = _build_pipeline(images, EncodeVAE(in_name="image", out_name="latent", vae=tiny_vae))
        recording = RecordingVAE(tiny_vae)
        batched = _build_pipeline(
            images,
            EncodeVAE(in_name="image", out_name="latent", vae=recording, batch_collector=True, max_batch_size=4),
        )

        with torch.no_grad():
            refs = [baseline.get_item(0, i)["latent"] for i in range(NUM_ITEMS)]
        results = _concurrent_results(batched, NUM_ITEMS)

        assert max(shape[0] for shape in recording.batch_shapes) > 1, "nothing was batched"
        assert sum(shape[0] for shape in recording.batch_shapes) == NUM_ITEMS
        for i in range(NUM_ITEMS):
            assert results[i].parameters.shape == refs[i].parameters.shape
            torch.testing.assert_close(results[i].parameters, refs[i].parameters, atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(results[i].mode(), refs[i].mode(), atol=1e-5, rtol=1e-4)
            params = results[i].parameters
            assert params.untyped_storage().nbytes() == params.numel() * params.element_size()

    def test_pixel_budget_caps_batch(self, tiny_vae, images):
        recording = RecordingVAE(tiny_vae)
        # 2048 px: two 32x32 images per batch, four of the 16x32 ones.
        batched = _build_pipeline(
            images,
            EncodeVAE(
                in_name="image",
                out_name="latent",
                vae=recording,
                batch_collector=True,
                max_batch_size=8,
                max_batch_pixels=2048,
            ),
        )

        _concurrent_results(batched, NUM_ITEMS)

        for shape in recording.batch_shapes:
            assert shape[0] * shape[2] * shape[3] <= 2048
        assert sum(shape[0] for shape in recording.batch_shapes) == NUM_ITEMS
