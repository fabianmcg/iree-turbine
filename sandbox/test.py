import torch
import torch.nn as nn

from run import TorchModule, IREEModule, from_torch, rel_error


def v1(x, y):
    class Test(nn.Module):
        def forward(self, x, y):
            return x * y, x + y

    model = Test()
    iree_run = IREEModule.from_torch(model, (x, y))
    torch_run = TorchModule(model, (x, y))
    torch_r = torch_run.run()
    iree_r = iree_run.run()
    torch.cuda.synchronize()

    for i, (t_r, i_r) in enumerate(zip(torch_r, iree_r)):
        print(f"Error out[{i}]: {rel_error(t_r, i_r)}", flush=True)

    return torch_r, iree_r[0]


def v2(x, y):
    class Test(nn.Module):
        def forward(self, x, y):
            return x + y, x * y

    model = Test()
    iree_run = IREEModule.from_torch(model, (x, y))
    torch_run = TorchModule(model, (x, y))
    torch_r = torch_run.run()
    iree_r = iree_run.run()
    torch.cuda.synchronize()

    for i, (t_r, i_r) in enumerate(zip(torch_r, iree_r)):
        print(f"Error out[{i}]: {rel_error(t_r, i_r)}", flush=True)

    return iree_r[0]


def main():
    x = torch.randn((8, 512, 512)).cuda()
    y = torch.randn((8, 512, 512)).cuda()

    t_t, i_m = v1(x, y)
    i_s = v2(x, y)

    for i, (t_r, i_r) in enumerate(zip(t_t, (i_m, i_s))):
        print(f"Error out[{i}]: {rel_error(t_r, i_r)}", flush=True)


if __name__ == "__main__":
    main()
