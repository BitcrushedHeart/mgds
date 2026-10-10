from random import Random

from mgds.pipelineModules.AspectBatchSorting import AspectBatchSorting


class SortingHarness(AspectBatchSorting):
    def __init__(self, resolutions, batch_size):
        super().__init__(resolution_in_name="crop_resolution", names=["row"], batch_size=batch_size)
        self.resolutions = resolutions
        self.current_variation = 0

    def _get_previous_length(self, name):
        return len(self.resolutions)

    def _get_previous_item(self, variation, name, index):
        if name == "crop_resolution":
            return self.resolutions[index]
        return index

    def _get_rand(self, variation, index=None):
        return Random(variation)


def test_exact_crop_buckets_keep_only_complete_effective_batches():
    sorting = SortingHarness(
        [(768, 768)] * 10 + [(512, 512)] * 9 + [(256, 256)] * 4,
        batch_size=5,
    )

    sorting.start(0)

    assert sorting.length() == 15
    kept_resolutions = [sorting.resolutions[index] for index in sorting.index_list]
    assert kept_resolutions.count((768, 768)) == 10
    assert kept_resolutions.count((512, 512)) == 5
    assert kept_resolutions.count((256, 256)) == 0


def test_world_size_effective_batch_drops_incomplete_repeated_bucket():
    # Repeats/balancing have already expanded rows before this module. For a
    # per-device batch of 3 across two workers, the effective bucket batch is 6.
    sorting = SortingHarness([(256, 256)] * 11 + [(128, 128)] * 6, batch_size=6)

    sorting.start(0)

    assert sorting.length() == 12
    kept_resolutions = [sorting.resolutions[index] for index in sorting.index_list]
    assert kept_resolutions.count((256, 256)) == 6
    assert kept_resolutions.count((128, 128)) == 6
