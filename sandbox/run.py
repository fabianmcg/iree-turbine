import numpy as np
import torch
import torch.nn as nn
import iree.turbine.aot as aot

from iree.compiler import compile_file, OutputFormat
from iree.runtime import (
    VmModule,
    DeviceArray,
    asdevicearray,
    BufferUsage,
    HalDevice,
)
from iree.turbine.runtime import Launchable, Device

from tempfile import TemporaryDirectory
from torch.autograd import DeviceType
from torch.profiler import profile as torch_profile, ProfilerActivity
from pathlib import Path

from typing import Sequence, Any, Optional


def _compile_exported(exported: aot.ExportOutput, **kwargs):
    buffer = None
    with TemporaryDirectory() as tmp:
        exported_name = Path(tmp) / "exported.mlirbc"
        exported.save_mlir(str(exported_name))
        buffer = compile_file(str(exported_name), **kwargs)

    assert buffer is not None
    return buffer


def from_torch(v: Any):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    return v


def rel_error(x_true: np.ndarray | torch.Tensor, x: np.ndarray | torch.Tensor):
    assert type(x_true) == type(x)
    if isinstance(x_true, torch.Tensor):
        return torch.linalg.norm(x - x_true) / torch.linalg.norm(x_true)
    x_true = x_true.astype(np.float64)
    x = x.astype(np.float64)
    return np.linalg.norm(x - x_true) / np.linalg.norm(x_true)


class Run:
    def __init__(self, arguments: Sequence[Any]):
        self.arguments = list(arguments)

    def run(self):
        pass

    def profile(
        self, num_its: int = 10, print_profile: bool = True, row_limit: int = 20
    ):
        with torch_profile(
            activities=[ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=5, active=num_its),
            record_shapes=True,
        ) as prof:
            for i in range(num_its + 6):
                self.run()
                prof.step()
        events = prof.key_averages(group_by_input_shape=True)
        if print_profile:
            print(
                events.table(sort_by="self_cuda_time_total", row_limit=row_limit),
                flush=True,
            )
        return np.array(
            [
                event.self_device_time_total
                for event in events
                if event.device_type == DeviceType.CUDA
            ]
        ).sum() / (num_its * 1000.0)


class TorchModule(Run):
    def __init__(self, module: nn.Module, arguments: Sequence[Any]):
        super().__init__(arguments)
        self.module = module

    def run(self):
        return self.module(*self.arguments)


class IREEModule(Run):
    def __init__(self, vmfb_bytes: bytes, arguments: Sequence[Any]):
        super().__init__(list(arguments))

        def get_vmfb(device: Device):
            vm_instance = device.vm_instance
            return VmModule.copy_buffer(vm_instance, vmfb_bytes)

        self.kernel = Launchable.from_vm_module(get_vmfb)

    @staticmethod
    def compile(
        module: nn.Module,
        arguments: Sequence[Any],
        intermediate_folder: Optional[str] = None,
        **kwargs,
    ):
        exported = aot.export(
            module,
            args=(*arguments,),
            import_symbolic_shape_expressions=True,
            **kwargs,
        )
        if intermediate_folder is not None:
            folder = Path(intermediate_folder)
            folder.mkdir(parents=True, exist_ok=True)
            with open(folder / "aot.mlir", "w") as file:
                print(exported.mlir_module, file=file)
        else:
            print(exported.mlir_module, flush=True)
        vmfb_bytes = _compile_exported(
            exported,
            target_backends=["rocm"],
            optimize=True,
            extra_args=[
                "--iree-hip-target=gfx942",
                "--iree-opt-level=O3",
                "--iree-opt-strip-assertions=true",
                "--iree-preprocessing-pass-pipeline=builtin.module(util.func(iree-preprocessing-make-single-dispatch))",
            ],
            output_format=OutputFormat.FLATBUFFER_BINARY,
            strip_source_map=True,
            strip_debug_ops=True,
            output_mlir_debuginfo=False,
        )
        return vmfb_bytes

    @staticmethod
    def from_torch(
        module: nn.Module,
        arguments: Sequence[Any],
        intermediate_folder: Optional[str] = None,
        **kwargs,
    ):
        vmfb_bytes = IREEModule.compile(
            module, arguments, intermediate_folder=intermediate_folder, **kwargs
        )
        return IREEModule(vmfb_bytes, [a for a in arguments])

    def run(self):
        return self.kernel(*self.arguments)
