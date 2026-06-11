import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Define shapes and strides for as_strided_ tests
AS_STRIDED_SHAPES = [(12,), (24,), (48,), (120,)]
AS_STRIDED_SIZE_STRIDE_COMBOS = [
    # (size, stride)
    ([3, 4], [4, 1]),
    ([4, 3], [3, 1]),
    ([2, 6], [6, 1]),
    ([6, 2], [2, 1]),
    ([12, 1], [1, 1]),
    ([1, 12], [12, 1]),
    ([2, 2, 3], [6, 3, 1]),
]


@pytest.mark.as_strided_
@pytest.mark.parametrize("shape", AS_STRIDED_SHAPES)
@pytest.mark.parametrize("size_stride", AS_STRIDED_SIZE_STRIDE_COMBOS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_as_strided_(shape, size_stride, dtype):
    """Test as_strided_ operator changes shape and stride correctly."""
    size, stride = size_stride

    # Skip if size product doesn't match shape product
    if np.prod(size) != np.prod(shape):
        pytest.skip("Size product doesn't match shape product")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    # Apply as_strided_
    ref_out = torch.as_strided_(ref_inp, size, stride)
    with flag_gems.use_gems():
        res_out = torch.as_strided_(inp, size, stride)

    # Verify shape and stride match
    assert res_out.shape == torch.Size(
        size
    ), f"Shape mismatch: {res_out.shape} vs {size}"
    assert res_out.stride() == tuple(
        stride
    ), f"Stride mismatch: {res_out.stride()} vs {stride}"

    # Verify data is correct
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.as_strided_
@pytest.mark.parametrize("shape", AS_STRIDED_SHAPES)
@pytest.mark.parametrize("size_stride", AS_STRIDED_SIZE_STRIDE_COMBOS)
@pytest.mark.parametrize("storage_offset", [0, 2, 4, 6])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_as_strided_with_offset(shape, size_stride, storage_offset, dtype):
    """Test as_strided_ with storage_offset parameter."""
    size, stride = size_stride

    # Skip if size product + offset exceeds original size
    if np.prod(size) + storage_offset > np.prod(shape):
        pytest.skip("Size + offset exceeds original size")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    # Apply as_strided_ with storage_offset
    ref_out = torch.as_strided_(ref_inp, size, stride, storage_offset)
    with flag_gems.use_gems():
        res_out = torch.as_strided_(inp, size, stride, storage_offset)

    # Verify shape and stride match
    assert res_out.shape == torch.Size(
        size
    ), f"Shape mismatch: {res_out.shape} vs {size}"
    assert res_out.stride() == tuple(
        stride
    ), f"Stride mismatch: {res_out.stride()} vs {stride}"

    # Verify data is correct
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.as_strided_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_as_strided_inplace(dtype):
    """Test that as_strided_ is in-place (modifies original tensor)."""
    # Hardcoded shape for in-place identity verification test
    shape = (12,)
    size = [3, 4]
    stride = [4, 1]

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    # Apply as_strided_
    ref_out = ref_inp.as_strided_(size, stride)
    with flag_gems.use_gems():
        res_out = inp.as_strided_(size, stride)

    # Verify it's the same object (in-place)
    assert res_out is inp, "as_strided_ should return same tensor object"
    assert ref_out is ref_inp, "Reference as_strided_ should return same tensor object"

    # Verify shape
    assert inp.shape == torch.Size(size)

    # Verify data is accessible through both views
    utils.gems_assert_close(inp, ref_inp, dtype)
