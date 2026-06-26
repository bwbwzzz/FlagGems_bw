import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.arctan2
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arctan2(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.arctan2(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = torch.arctan2(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.arctan2
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arctan2_special_values(dtype):
    x = torch.tensor(
        [
            0.0,
            float("inf"),
            -float("inf"),
            1.0,
            1.0,
            float("nan"),
            1.0,
        ],
        dtype=dtype,
        device=flag_gems.device,
    )
    y = torch.tensor(
        [
            0.0,
            1.0,
            1.0,
            float("inf"),
            -float("inf"),
            1.0,
            float("nan"),
        ],
        dtype=dtype,
        device=flag_gems.device,
    )

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.arctan2(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = torch.arctan2(x, y)

    # NaN cannot be compared by the default gems_assert_close,
    # because equal_nan=False by default.
    nan_mask = torch.isnan(ref_out)
    assert torch.equal(torch.isnan(res_out), nan_mask)

    utils.gems_assert_close(res_out[~nan_mask], ref_out[~nan_mask], dtype)


@pytest.mark.arctan2
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arctan2_broadcast(dtype):
    x = torch.randn((4, 1), dtype=dtype, device=flag_gems.device)
    y = torch.randn((1, 4), dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.arctan2(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = torch.arctan2(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)
