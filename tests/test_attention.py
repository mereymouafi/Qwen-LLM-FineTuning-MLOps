import torch
from src.copilot.attention import scaled_dot_product_attention

def test_attention_weights_sum_to_one():
    x = torch.randn(2, 3, 4)
    out, weights = scaled_dot_product_attention(x, x, x)
    assert out.shape == x.shape
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 3), atol=1e-6)
