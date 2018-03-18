import os


def set_gpu_recursive(var, gpu_id):
    for key in var:
        if isinstance(var[key], dict):
            var[key] = set_gpu_recursive(var[key], gpu_id)
        else:
            try:
                var[key] = var[key].cuda(gpu_id)
            except:
                pass
    return var


def get_last_epoch(path):
    last_epoch = 0
    for r, d, f in os.walk(os.path.join(path, 'checkpoints')):
        for ckpt_dir in d:
            epoch = int(ckpt_dir)
            if epoch > last_epoch:
                last_epoch = epoch
    return last_epoch
