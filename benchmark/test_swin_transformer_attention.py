import pytest
import torch

import flag_gems

from . import base, consts

# Swin Transformer Attention benchmark shapes
# (B, H, S, D) format: batch, heads, sequence_length, head_dim
SWIN_ATTENTION_SHAPES = [
    (1, 2, 16, 32),
    (2, 4, 32, 64),
    (4, 8, 64, 128),
]


class SwinTransformerAttentionBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SWIN_ATTENTION_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            B, H, S, D = shape
            query = torch.randn(B, H, S, D, dtype=cur_dtype, device=self.device)
            key = torch.randn(B, H, S, D, dtype=cur_dtype, device=self.device)
            value = torch.randn(B, H, S, D, dtype=cur_dtype, device=self.device)
            yield query, key, value


@pytest.mark.swin_transformer_attention
def test_swin_transformer_attention():
    bench = SwinTransformerAttentionBenchmark(
        op_name="swin_transformer_attention",
        torch_op=torch.nn.functional.scaled_dot_product_attention,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.Swin_Transformer_Attention)
    bench.run()
