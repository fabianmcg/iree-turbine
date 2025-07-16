import torch
import torch.nn as nn

from run import TorchModule, IREEModule, from_torch, rel_error


def main():
    class CausalAttention(nn.Module):
        def forward(self, q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )

    model = CausalAttention()
    bs = 4
    num_heads = 16
    seq_len = 1011
    head_dim = 64

    q = torch.randn((bs, num_heads, 1, head_dim)).cuda()
    k = torch.randn((bs, num_heads, seq_len, head_dim)).cuda()
    v = torch.randn((bs, num_heads, seq_len, head_dim)).cuda()
    dyn_seq_len = torch.export.Dim("seq_len")

    iree_run = IREEModule.from_torch(
        model,
        (q, k, v),
        dynamic_shapes={
            "q": {},  # causal or prefill static
            "k": {2: dyn_seq_len},
            "v": {2: dyn_seq_len},
        },
    )

    torch_run = TorchModule(model, (q, k, v))

    tet = torch_run.profile()
    iet = iree_run.profile()
    print(f"Total torch time: {tet:.3f} ms", flush=True)
    print(f"Total IREE time: {iet:.3f} ms", flush=True)
    print(f"Numeric error: {rel_error(torch_run.run(), iree_run.run())}")


if __name__ == "__main__":
    main()
