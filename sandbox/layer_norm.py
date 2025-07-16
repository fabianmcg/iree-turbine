import torch
import torch.nn as nn
from pathlib import Path

from run import TorchModule, IREEModule, from_torch, rel_error

torch.manual_seed(0)


def aten(arguments: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    print("*" * 80, flush=True)
    print("torch.aten:", flush=True)

    class Module(nn.Module):
        def forward(self, in_t: torch.Tensor, w_t: torch.Tensor, b_t: torch.Tensor):
            return torch.ops.aten.native_layer_norm(in_t, w_t.shape, w_t, b_t, 1e-05)

    model = Module().cuda()

    torch_run = TorchModule(model, arguments)
    iree_run = IREEModule.from_torch(model, arguments)

    tet = torch_run.profile()
    iet = iree_run.profile()

    print(f"Total torch time: {tet:.3f} ms", flush=True)
    print(f"Total IREE time: {iet:.3f} ms", flush=True)

    torch_r = torch_run.run()
    iree_r = iree_run.run()

    for i, (t_r, i_r) in enumerate(zip(torch_r, iree_r)):
        print(f"Error out[{i}]: {rel_error(t_r, i_r)}", flush=True)


def regular(arguments: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    print("*" * 80, flush=True)
    print("torch.nn.functional:", flush=True)

    class Module(nn.Module):
        def forward(self, in_t: torch.Tensor, w_t: torch.Tensor, b_t: torch.Tensor):
            return torch.nn.functional.layer_norm(in_t, w_t.shape, w_t, b_t, 1e-05)

    model = Module().cuda()

    iree_run = IREEModule.from_torch(model, arguments)
    torch_run = TorchModule(model, arguments)

    tet = torch_run.profile()
    iet = iree_run.profile()

    print(f"Total torch time: {tet:.3f} ms", flush=True)
    print(f"Total IREE time: {iet:.3f} ms", flush=True)
    print(f"Error: {rel_error(torch_run.run(), iree_run.run())}")


def main():
    arguments = (
        torch.randn((16, 256, 2048), dtype=torch.float).cuda(),
        torch.randn((2048,), dtype=torch.float).cuda(),
        torch.randn((2048,), dtype=torch.float).cuda(),
    )

    aten(arguments)
    regular(arguments)


if __name__ == "__main__":
    main()
