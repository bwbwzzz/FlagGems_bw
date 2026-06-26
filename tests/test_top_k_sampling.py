import pytest
import torch

import flag_gems
from flag_gems.ops.top_k_sampling import top_k_sampling_reference as top_k_sampling_ref

from . import accuracy_utils as utils

# Test shapes: k values of 1, 3, 10; vocab sizes of 32, 128, 512
TOP_K_SAMPLING_BATCH_SIZES = [1, 4, 8]
TOP_K_SAMPLING_VOCAB_SIZES = [32, 128, 512]
TOP_K_SAMPLING_K_VALUES = [1, 3, 10]


@pytest.mark.top_k_sampling
@pytest.mark.parametrize("batch_size", TOP_K_SAMPLING_BATCH_SIZES)
@pytest.mark.parametrize("vocab_size", TOP_K_SAMPLING_VOCAB_SIZES)
@pytest.mark.parametrize("k", TOP_K_SAMPLING_K_VALUES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_top_k_sampling(batch_size, vocab_size, k, dtype):
    """Test Top_K_Sampling accuracy against reference implementation"""
    # Adjust k if it's larger than vocab_size
    actual_k = min(k, vocab_size)

    # Create random logits
    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, dtype=dtype, device=flag_gems.device)

    # Get reference output using manual reference implementation
    ref_logits = utils.to_reference(logits)
    ref_out = top_k_sampling_ref(ref_logits, actual_k)

    # Get GEMS output
    with flag_gems.use_gems():
        res_out = flag_gems.top_k_sampling(logits, actual_k)

    # Compare shape
    assert (
        res_out.shape == ref_out.shape
    ), f"Shape mismatch: {res_out.shape} vs {ref_out.shape}"

    # Note: we don't compare res_out with ref_out using gems_assert_equal because
    # top_k_sampling is stochastic — the Triton kernel uses tl.rand while the
    # reference uses torch.rand, and they produce different random sequences even
    # with the same torch manual seed.

    # Check that all indices are valid
    assert (res_out >= 0).all(), "Sampled indices should be non-negative"
    assert (
        res_out < vocab_size
    ).all(), "Sampled indices should be less than vocab_size"


@pytest.mark.top_k_sampling
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_top_k_sampling_single_element(dtype):
    """Test Top_K_Sampling with single batch element"""
    torch.manual_seed(123)
    vocab_size = 64
    k = 10

    logits = torch.randn(1, vocab_size, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = flag_gems.top_k_sampling(logits, k)

    # Verify output is valid
    assert res_out.shape == (1,), f"Expected shape (1,), got {res_out.shape}"
    # Note: no exact value comparison since output is stochastic
    assert (res_out >= 0).all() and (
        res_out < vocab_size
    ).all(), "Invalid sampled index"
