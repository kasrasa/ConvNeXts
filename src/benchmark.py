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


def inference_time(model, input_tensor, device, num_iterations=100):
    """Measure average inference time per iteration for a model on a device.

    The input_tensor should be preloaded to the same device as the model.
    """
    model.to(device)
    model.eval()
    flop_counter = FlopCounterMode(model, input_tensor)

    with torch.inference_mode():
        for _ in range(10):
            _ = model(input_tensor)
    with flop_counter:
        if device == "mps":
            torch.mps.synchronize()

            start_time = time.time()
            with torch.inference_mode():
                for _ in range(num_iterations):
                    _ = model(input_tensor)
            end_time = time.time()

            elapsed_time = end_time - start_time
            avg_time_per_iteration = elapsed_time / num_iterations
        elif device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            with torch.inference_mode():
                for _ in range(num_iterations):
                    _ = model(input_tensor)
            end_event.record()
            torch.cuda.synchronize()

            elapsed_time = start_event.elapsed_time(end_event) / 1000.0
            avg_time_per_iteration = elapsed_time / num_iterations
        elif device == "cpu":
            start_time = time.time()
            with torch.inference_mode():
                for _ in range(num_iterations):
                    _ = model(input_tensor)
            end_time = time.time()

            elapsed_time = end_time - start_time
            avg_time_per_iteration = elapsed_time / num_iterations
        else:
            raise ValueError(f"Unsupported device: {device}")

    total_flops = flop_counter.total_flops()

    return avg_time_per_iteration, total_flops