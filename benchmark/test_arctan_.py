import pytest
import torch

from . import base, consts


@pytest.mark.arctan_
def test_arctan_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="arctan_",
        torch_op=torch.arctan_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
