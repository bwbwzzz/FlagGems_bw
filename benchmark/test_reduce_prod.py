import pytest
import torch

from . import base, consts


@pytest.mark.reduce_prod
def test_reduce_prod():
    bench = base.UnaryReductionBenchmark(
        op_name="reduce_prod", torch_op=torch.prod, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
