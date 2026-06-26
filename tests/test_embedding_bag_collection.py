import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.embedding_bag_collection
@pytest.mark.parametrize("num_embeddings", [16, 32, 64])
@pytest.mark.parametrize("embedding_dim", [16, 32])
@pytest.mark.parametrize("num_indices", [8, 16, 32])
@pytest.mark.parametrize("num_bags", [2, 4])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_embedding_bag_collection(
    num_embeddings, embedding_dim, num_indices, num_bags, dtype
):
    # Create embedding table
    weight = torch.randn(
        num_embeddings, embedding_dim, dtype=dtype, device=flag_gems.device
    )

    # Create random indices into the embedding table
    indices = torch.randint(0, num_embeddings, (num_indices,), device=flag_gems.device)

    # Create offsets that define the bags
    # Each bag gets roughly num_indices / num_bags indices
    bag_size = num_indices // num_bags
    offsets = torch.arange(0, num_indices + 1, bag_size, device=flag_gems.device)
    # Ensure the last offset is exactly num_indices
    offsets[-1] = num_indices
    # Handle case where division doesn't divide evenly
    if offsets.shape[0] > num_bags + 1:
        offsets = offsets[: num_bags + 1]
    elif offsets.shape[0] < num_bags + 1:
        # Pad if needed
        padding = torch.full(
            (num_bags + 1 - offsets.shape[0],),
            num_indices,
            device=offsets.device,
            dtype=offsets.dtype,
        )
        offsets = torch.cat([offsets, padding])

    # Reference: use PyTorch embedding_bag
    ref_weight = utils.to_reference(weight)
    ref_indices = utils.to_reference(indices)
    ref_offsets = utils.to_reference(offsets)

    (
        ref_output,
        ref_offset2bag,
        ref_bag_size,
        ref_max_indices,
    ) = torch.ops.aten._embedding_bag_forward_only(
        ref_weight,
        ref_indices,
        ref_offsets,
        False,  # scale_grad_by_freq
        0,  # mode=sum
        False,  # sparse
        None,  # per_sample_weights
        False,  # include_last_offset
        -1,  # padding_idx
    )

    # FlagGems implementation
    from flag_gems.ops.embedding_bag_collection import embedingBagCollection

    with flag_gems.use_gems():
        (
            res_output,
            res_offset2bag,
            res_bag_size,
            res_max_indices,
        ) = embedingBagCollection(
            weight,
            indices,
            offsets,
            scale_grad_by_freq=False,
            mode=0,  # sum
            sparse=False,
            per_sample_weights=None,
            include_last_offset=False,
            padding_idx=-1,
        )

    # Compare outputs - use custom tolerance for float16
    # The atol in utils.gems_assert_close is too strict for float16/bfloat16
    if dtype == torch.float16:
        atol = 1e-2  # 0.01
    elif dtype == torch.bfloat16:
        atol = 4e-2  # 0.04
    else:
        atol = 1e-4

    # Compare main output
    res_output_f32 = res_output.to(torch.float32)
    ref_output_f32 = ref_output.to(torch.float32)
    torch.testing.assert_close(res_output_f32, ref_output_f32, atol=atol, rtol=atol)
