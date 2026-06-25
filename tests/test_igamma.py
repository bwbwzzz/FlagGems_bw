import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.igamma
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_igamma(shape, dtype):
    # Ensure positive inputs for igamma (lower incomplete gamma function)
    inp1 = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
    inp2 = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.igamma(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.igamma(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.igamma_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_igamma_(shape, dtype):
    # Ensure positive inputs for igamma_
    inp1 = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
    inp2 = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1

    ref_inp1 = utils.to_reference(inp1.clone(), True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = ref_inp1.igamma_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.igamma_(inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
