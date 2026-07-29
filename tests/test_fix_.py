import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.fix_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fix_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 10
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = ref_inp.fix_()
    with flag_gems.use_gems():
        res_out = inp.fix_()

    utils.gems_assert_close(res_out, ref_out, dtype)
