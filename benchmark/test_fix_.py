import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.fix_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_fix_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="fix_",
        torch_op=lambda x: torch.fix_(x),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
