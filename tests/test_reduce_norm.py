import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

KEEPDIM_DIMS = list(zip([True, False] * 2, [0, 1, [0, 1], [1, 0]]))


@pytest.mark.reduce_norm
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("ord", [2, float("inf"), -float("inf"), 0, 1])
@pytest.mark.parametrize("keepdim, dim", KEEPDIM_DIMS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reduce_norm(shape, ord, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    # Compare against torch.linalg.vector_norm as reference
    ref_out = torch.linalg.vector_norm(ref_inp, ord, dim, keepdim)
    with flag_gems.use_gems():
        res_out = flag_gems.reduce_norm(inp, ord, dim, keepdim)

    utils.gems_assert_close(res_out, ref_out, dtype)
