import pytest
import torch

from . import base, consts


@pytest.mark.divide
def test_divide():
    bench = base.BinaryPointwiseBenchmark(
        op_name="divide",
        torch_op=torch.divide,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
