"""Explicit integration-only SGD identity digest; disabled without audit steps."""
import hashlib
import json


def record_parameters(torch, stage, output, rank, iteration):
    digest = hashlib.sha256()
    tensors = elements = 0
    finite = True
    for name, parameter in stage.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update((name + ':' + str(value.dtype) + ':' + str(list(value.shape)) + '\n').encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
        tensors += 1
        elements += value.numel()
        finite &= bool(torch.isfinite(value).all())
    row = dict(rank=rank, iteration=iteration, parameter_sha256=digest.hexdigest(),
               parameter_tensors=tensors, parameter_elements=elements, all_finite=finite)
    with (output / f'parameter_numerical_rank{rank}.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
