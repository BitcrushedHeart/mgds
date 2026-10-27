from contextlib import nullcontext

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule
from mgds.TextEncoderBatching import BatchCollector, PendingEncode

import torch

from diffusers import AutoencoderDC, AutoencoderKL, AutoencoderKLFlux2, AutoencoderKLQwenImage
from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import AutoencoderKLHunyuanVideo


class EncodeVAE(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
        self,
        in_name: str,
        out_name: str,
        vae: AutoencoderKL | AutoencoderDC | AutoencoderKLHunyuanVideo | AutoencoderKLQwenImage | AutoencoderKLFlux2,
        autocast_contexts: list[torch.autocast | None] = None,
        dtype: torch.dtype | None = None,
        sample_mode: str = "sample",
        batch_collector: bool = False,
        max_batch_size: int = 8,
        max_batch_pixels: int | None = None,
    ):
        super().__init__()
        self.in_name = in_name
        self.out_name = out_name
        self.vae = vae

        self.autocast_contexts = [nullcontext()] if autocast_contexts is None else autocast_contexts
        self.dtype = dtype

        self._batch_collector = None
        if batch_collector:
            self.enable_batch_collector(max_batch_size, max_batch_pixels)

    def enable_batch_collector(self, max_batch_size: int, max_batch_pixels: int | None = None):
        """Encode concurrent requests of the same shape as one batched forward.

        Only images of identical shape, dtype and device can be stacked, so
        requests are grouped on those; ``max_batch_pixels`` caps each batch's
        total pixels (frames included for video) to bound activation memory.
        Meant for cache builds, where many worker threads encode at once.
        """
        if max_batch_size <= 1:
            self._batch_collector = None
            return

        def group_key(request: PendingEncode):
            image = request.tokens
            return tuple(image.shape), image.dtype, image.device

        def batch_limit(request: PendingEncode) -> int:
            if max_batch_pixels is None:
                return max_batch_size
            image = request.tokens
            return max_batch_pixels // max(1, image.numel() // image.shape[0])

        self._batch_collector = BatchCollector(
            self._run_collected_batch,
            max_batch_size,
            group_key_fun=group_key,
            batch_limit_fun=batch_limit,
        )

    def length(self) -> int:
        return self._get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def _encode(self, images: torch.Tensor) -> list:
        # With high concurrency, we can occasionally get memory spikes here
        # that prevent allocation even if we're nowhere near the memory limit
        # of the GPU. So try a few times.
        retries = 0
        while True:
            try:
                with self._all_contexts(self.autocast_contexts):
                    vae_output = self.vae.encode(images)
                break
            except RuntimeError:
                retries += 1
                if retries > 3:
                    raise

        if hasattr(vae_output, "latent_dist"):
            distribution = vae_output.latent_dist
            if images.shape[0] == 1:
                return [distribution]
            # Split back into the per-item, batch-of-one distributions the
            # unbatched path returns. Clone, don't view: torch.save writes a
            # view's whole storage, so every cached item would carry the batch.
            return [
                type(distribution)(distribution.parameters[i : i + 1].clone(), deterministic=distribution.deterministic)
                for i in range(images.shape[0])
            ]
        if hasattr(vae_output, "latent"):
            return [latent.clone() for latent in vae_output.latent.unbind(dim=0)]
        raise ValueError(f"unsupported VAE output type {type(vae_output).__name__}")

    def _run_collected_batch(self, batch: list[PendingEncode]):
        outputs = self._encode(torch.stack([request.tokens for request in batch]))
        for request, output in zip(batch, outputs, strict=True):
            request.result = output

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        image = self._get_previous_item(variation, self.in_name, index)

        if self.dtype:
            image = image.to(dtype=self.dtype)

        if self._batch_collector is not None:
            output = self._batch_collector.encode(image, None)
        else:
            output = self._encode(image.unsqueeze(0))[0]

        return {self.out_name: output}
