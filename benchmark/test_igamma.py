import pytest
import torch

from . import base


@pytest.mark.igamma
def test_igamma():
    bench = base.BinaryPointwiseBenchmark(
        op_name="igamma",
        torch_op=torch.igamma,
        # CUDA does not support igamma for Half/BFloat16
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.igamma_
def test_igamma_():
    # torch.igamma_ is not a standalone function; use lambda wrapper for inplace method
    def igamma_inplace(a, x):
        return a.igamma_(x)

    bench = base.BinaryPointwiseBenchmark(
        op_name="igamma_",
        torch_op=igamma_inplace,
        # CUDA does not support igamma_ for Half/BFloat16
        dtypes=[torch.float32],
    )
    bench.run()
