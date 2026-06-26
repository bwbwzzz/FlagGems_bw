import pytest
import torch

from . import base

# Worktree original: embedingBagCollection benchmark
# Note: bfloat16 not supported by underlying embedding_bag kernel
EMBEDDING_BAG_SHAPES = [
    (16, 16, 8, 2),  # num_embeddings, embedding_dim, num_indices, num_bags
    (32, 32, 16, 4),
    (64, 64, 32, 8),
    (128, 128, 64, 16),
    (256, 256, 128, 32),
]


class EmbeddingBagCollectionBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = EMBEDDING_BAG_SHAPES

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            num_embeddings, embedding_dim, num_indices, num_bags = shape
            weight = torch.randn(
                num_embeddings, embedding_dim, dtype=dtype, device=self.device
            )
            indices = torch.randint(
                0, num_embeddings, (num_indices,), device=self.device
            )
            bag_size = num_indices // num_bags
            offsets = torch.arange(0, num_indices + 1, bag_size, device=self.device)
            offsets[-1] = num_indices
            if offsets.shape[0] > num_bags + 1:
                offsets = offsets[: num_bags + 1]
            elif offsets.shape[0] < num_bags + 1:
                padding = torch.full(
                    (num_bags + 1 - offsets.shape[0],),
                    num_indices,
                    device=offsets.device,
                    dtype=offsets.dtype,
                )
                offsets = torch.cat([offsets, padding])
            yield weight, indices, offsets


@pytest.mark.embedding_bag_collection
def test_embedding_bag_collection():
    from flag_gems.ops.embedding_bag_collection import embedingBagCollection

    bench = EmbeddingBagCollectionBenchmark(
        op_name="embedding_bag_collection",
        torch_op=embedingBagCollection,
        # Note: bfloat16 is not supported by embedding bag Triton kernel
        dtypes=[torch.float16, torch.float32],
    )
    bench.run()
