import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.top_k_sampling
def test_top_k_sampling():
    def top_k_sampling_input_fn(shape, dtype, device):
        batch_size, vocab_size = shape
        # k should be less than vocab_size
        k = min(10, vocab_size)
        logits = torch.randn(shape, dtype=dtype, device=device)
        yield logits, k

    bench = base.GenericBenchmark2DOnly(
        input_fn=top_k_sampling_input_fn,
        op_name="top_k_sampling",
        torch_op=flag_gems.top_k_sampling,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
