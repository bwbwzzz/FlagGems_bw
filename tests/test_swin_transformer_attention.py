import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Swin Transformer Attention test shapes
# (B, H, S, D) format: batch, heads, sequence_length, head_dim
ATTENTION_SHAPES = [
    (1, 2, 16, 32),
    (2, 4, 32, 64),
    (4, 8, 64, 128),
]


@pytest.mark.swin_transformer_attention
@pytest.mark.parametrize("shape", ATTENTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_swin_transformer_attention(shape, dtype):
    """Test Swin Transformer Attention accuracy against PyTorch reference"""
    B, H, S, D = shape

    # Skip unsupported head dimensions
    if D not in [16, 32, 64, 128, 256]:
        pytest.skip(f"Unsupported head dimension: {D}")

    # Create input tensors
    query = torch.randn(B, H, S, D, dtype=dtype, device=flag_gems.device)
    key = torch.randn(B, H, S, D, dtype=dtype, device=flag_gems.device)
    value = torch.randn(B, H, S, D, dtype=dtype, device=flag_gems.device)

    # Compute reference using scaled_dot_product_attention
    scale = 1.0 / (D**0.5)
    ref_query = utils.to_reference(query, True)
    ref_key = utils.to_reference(key, True)
    ref_value = utils.to_reference(value, True)
    ref_out = torch.nn.functional.scaled_dot_product_attention(
        ref_query, ref_key, ref_value, scale=scale
    )

    # Compute with GEMS
    with flag_gems.use_gems():
        res_out = flag_gems.Swin_Transformer_Attention(
            query, key, value, window_size=7, scale=scale
        )

    # Use appropriate tolerance based on dtype
    if dtype == torch.float32:
        atol = 1e-5
    elif dtype == torch.bfloat16:
        atol = 1e-2
    else:  # float16
        atol = 1e-3

    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol)
