import argparse
import re
import torch
import torch.nn as nn
from pathlib import Path

from util import TorchModule, IREEModule, rel_error, torch_types


def miopen(
    input_shape: tuple[int, ...],
    normalized_shape: tuple[int, ...],
    dtype: torch.dtype,
):
    from subprocess import run

    shape = "x".join(map(str, input_shape + normalized_shape))
    d_type = ""
    if dtype == torch.float16:
        d_type = "fp16"
    elif dtype == torch.float16:
        d_type = "bfp16"
    cmd = f"/opt/rocm/bin/MIOpenDriver layernorm{d_type} -F 1 -X {shape} -o {len(input_shape)} --time 1"
    print(cmd, flush=True)
    rr = run(cmd.split(" "), capture_output=True)
    rm = re.search(
        r"Elapsed.*", rr.stdout.decode("utf-8") + "\n" + rr.stderr.decode("utf-8")
    )
    if rm is not None:
        print(rm.group(), flush=True)
    else:
        print(rr.stdout.decode("utf-8") + "\n" + rr.stderr.decode("utf-8"), flush=True)


def layer_norm(
    input_shape: tuple[int, ...],
    normalized_shape: tuple[int, ...],
    dtype: torch.dtype,
    num_its: int,
    torch_compile: bool,
    eps: float = 1e-05,
):
    class Module(nn.Module):
        def forward(self, in_t: torch.Tensor, w_t: torch.Tensor, b_t: torch.Tensor):
            return torch.ops.aten.native_layer_norm(
                in_t, normalized_shape, w_t, b_t, eps
            )

    print("*" * 80, flush=True)
    print(
        f"Layer norm:\n\tinput_shape={input_shape}, normalized_shape={normalized_shape}\n"
        + f"\tdtype={dtype}, eps={eps}\n"
        + f"\ttorch_compile={torch_compile}",
        flush=True,
    )

    model = Module().cuda()
    if torch_compile:
        model_c = Module().cuda()

    arguments = (
        torch.randn(input_shape + normalized_shape, dtype=dtype).cuda(),
        torch.randn(normalized_shape, dtype=dtype).cuda(),
        torch.randn(normalized_shape, dtype=dtype).cuda(),
    )

    print("\n" + ("=" * 40), flush=True)
    print("Compiling IREE...", flush=True)
    iree_run = IREEModule.from_torch(model, arguments)
    print("Done compiling", flush=True)

    torch_run = TorchModule(model, arguments)
    print("\n" + ("=" * 40), flush=True)
    print("Profiling torch:", flush=True)
    tet = torch_run.profile(num_its)

    if torch_compile:
        model_c.compile()
        torch_c_run = TorchModule(model_c, arguments)
        print("Profiling torch compile:", flush=True)
        tetc = torch_c_run.profile(num_its)

    print("\n" + ("=" * 40), flush=True)
    print("Profiling IREE:", flush=True)
    iet = iree_run.profile(num_its)

    print("\n" + ("=" * 40), flush=True)
    print("MIOpen:", flush=True)
    miopen(input_shape, normalized_shape, dtype)

    print("\n" + ("=" * 40), flush=True)
    print("Summary:", flush=True)
    print(f"Total torch time: {tet:.3f} ms", flush=True)
    if torch_compile:
        print(f"Total torch compile time: {tetc:.3f} ms", flush=True)
    print(f"Total IREE time: {iet:.3f} ms", flush=True)

    torch_r = torch_run.run()
    iree_r = iree_run.run()
    for i, (t_r, i_r) in enumerate(zip(torch_r, iree_r)):
        print(f"Numeric error out[{i}]: {rel_error(t_r, i_r):.3e}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-shape",
        "-b",
        metavar="<batch shape>",
        nargs="*",
        type=int,
        default=(),
        help="batch dimensions",
    )
    parser.add_argument(
        "--normalized-shape",
        "-n",
        metavar="<normalized shape>",
        nargs="*",
        type=int,
        default=(),
        help="normalized dimensions",
    )
    parser.add_argument(
        "--num-its",
        "-i",
        metavar="<num its>",
        type=int,
        default=10,
        help="number of iterations to execute",
    )
    parser.add_argument(
        "--dtype", "-t", type=str, default="f16", choices=torch_types.keys()
    )
    parser.add_argument("-c", action="store_true", help="whether to use torch compile")
    args = parser.parse_args()
    layer_norm(
        input_shape=tuple(args.batch_shape),
        normalized_shape=tuple(args.normalized_shape),
        dtype=torch_types[args.dtype],
        num_its=args.num_its,
        torch_compile=args.c,
    )


if __name__ == "__main__":
    main()
