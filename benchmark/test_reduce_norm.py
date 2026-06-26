import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.reduce_norm
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_reduce_norm_perf(dtype):
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=base.unary_input_fn,
        op_name="reduce_norm",
        torch_op=torch.linalg.vector_norm,
        dtypes=[dtype],
    )
    bench.set_gems(flag_gems.reduce_norm)
    bench.run()
