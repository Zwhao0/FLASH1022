"""Explicit test-only full-gradient identity audit; never enabled in production."""
import hashlib
import json


def record_gradients(torch, stage, output, rank, iteration):
    digest = hashlib.sha256()
    tensors = elements = 0
    finite = True
    for name, parameter in stage.named_parameters():
        gradient = parameter.main_grad
        if gradient is None:
            digest.update((name + ':none\n').encode())
            continue
        value = gradient.detach().cpu().contiguous()
        digest.update((name + ':' + str(value.dtype) + ':' + str(list(value.shape)) + '\n').encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
        tensors += 1
        elements += value.numel()
        finite &= bool(torch.isfinite(value).all())
    row = dict(rank=rank, iteration=iteration, gradient_sha256=digest.hexdigest(),
               gradient_tensors=tensors, gradient_elements=elements, all_finite=finite)
    with (output / f'numerical_rank{rank}.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
