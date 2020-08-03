"""Code to compare optimizer step times and generate tracing timelines."""
import torch
from torch import nn
from torch.autograd import profiler
import time

from copy import deepcopy

from contiguous_params import ContiguousParams


def benchmark_model(model, optimizer, parameters, name):
    # Run 
    step_times = []
    # Autograd profiler adds some overhead, so we time the forward pass with
    # and without enabling it.
    for profile_autograd in [False, True]:
        with profiler.profile(enabled=profile_autograd, use_cuda=(device == "cuda")) as prof:
            for i in range(15):
                # Warm up for five steps, reset step_times after this.
                if i == 5:
                    step_times = []
                with profiler.record_function("forward"):
                    loss = model(x).sum()
                with profiler.record_function("backward"):
                    loss.backward()
                torch.cuda.synchronize()
                start = time.time()
                with profiler.record_function("gradient_norm"):
                    torch.nn.utils.clip_grad_norm_(parameters, 0.1)
                with profiler.record_function("step"):
                    optimizer.step()
                with profiler.record_function("zero_grad"):
                    optimizer.zero_grad()
                torch.cuda.synchronize()
                step_times.append(time.time() - start)
            print(f"Mean step time: {sum(step_times) / 10} seconds. "
                  f"(Autograd profiler enabled: {profile_autograd})")
    prof.export_chrome_trace(f"{name}_timeline.json")


if __name__ == "__main__":
    device = "cuda"
    model = nn.Sequential(*[nn.Linear(128, 128) for i in range(100)]).to(device)
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
    x = torch.randn(1, 128).to(device)

    model_copies = [deepcopy(model) for _ in range(2)]

    # Benchmark original.
    parameters = list(model_copies[0].parameters())
    optimizer = torch.optim.Adam(parameters)
    benchmark_model(model_copies[0], optimizer, parameters, "original_params")

    # Benchmark contiguous.
    parameters = ContiguousParams(model_copies[1].parameters())
    optimizer = torch.optim.Adam(parameters.contiguous())
    benchmark_model(model_copies[1], optimizer, parameters.contiguous(),
                    "contiguous_params")
    # Ensure the parameter buffers are still valid.
    parameters.assert_buffer_is_valid()

