import torch
import time
from torch.utils.flop_counter import FlopCounterMode


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train one epoch with the trainable parameters selected by the caller.

    Fully frozen submodules remain in eval mode, preventing BatchNorm
    statistics and stochastic layers from changing in frozen stages.
    """
    model.train()
    for module in model.modules():
        if module is model:
            continue
        parameters = tuple(module.parameters())
        if parameters and not any(param.requires_grad for param in parameters):
            module.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        correct_predictions += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    if total_samples == 0:
        raise ValueError("Training DataLoader is empty.")
    return running_loss / total_samples, correct_predictions / total_samples


def validate(model, dataloader, criterion, device):
    """Measure validation loss and accuracy without updating the model."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            correct_predictions += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    if total_samples == 0:
        raise ValueError("Validation DataLoader is empty.")
    return running_loss / total_samples, correct_predictions / total_samples


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_inference(model, input_tensor, device, num_iterations=100,
                        warmup_iterations=10, repeats=5):
    """Time forward-only batches; count FLOPs separately for one batch.

    Each sample is the average batch time over num_iterations, not an
    individual-request latency. Transfers and FLOP instrumentation are excluded.
    FLOPs cover operators registered by PyTorch; multiply-add counts as two.
    """
    import statistics

    device = torch.device(device)
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {device}")
    for name, value in (("num_iterations", num_iterations),
                        ("warmup_iterations", warmup_iterations), ("repeats", repeats)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if input_tensor.ndim == 0 or input_tensor.shape[0] < 1:
        raise ValueError("input_tensor must have a nonempty batch dimension.")
    model.to(device).eval()
    input_tensor = input_tensor.to(device)
    # Resolve an unindexed device (e.g. cuda) to the actual input device.
    device = input_tensor.device
    batch_size = input_tensor.shape[0]
    samples = []
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            model(input_tensor)
        for _ in range(repeats):
            _synchronize(device)
            start = time.perf_counter()
            for _ in range(num_iterations):
                model(input_tensor)
            _synchronize(device)
            samples.append((time.perf_counter() - start) / num_iterations)
        # One forward pass, outside every timed region.
        with FlopCounterMode(display=False) as counter:
            model(input_tensor)
        _synchronize(device)
    total_flops = int(counter.get_total_flops())
    median_seconds = statistics.median(samples)
    return {
        "device": str(device), "dtype": str(input_tensor.dtype),
        "input_shape": list(input_tensor.shape), "batch_size": batch_size,
        "num_iterations": num_iterations, "warmup_iterations": warmup_iterations,
        "repeats": repeats, "batch_seconds_samples": samples,
        "batch_latency_ms": median_seconds * 1000,
        "batch_latency_min_ms": min(samples) * 1000,
        "batch_latency_max_ms": max(samples) * 1000,
        "throughput_images_per_second": batch_size / median_seconds,
        "throughput_min_images_per_second": batch_size / max(samples),
        "throughput_max_images_per_second": batch_size / min(samples),
        "flops_per_batch": total_flops,
        "gflops_per_image": total_flops / batch_size / 1e9,
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def inference_time(model, input_tensor, device, num_iterations=100):
    """Compatibility wrapper returning seconds/batch and FLOPs for ONE batch."""
    result = benchmark_inference(model, input_tensor, device, num_iterations, repeats=1)
    return result["batch_seconds_samples"][0], result["flops_per_batch"]
