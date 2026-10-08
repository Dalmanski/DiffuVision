import gc
import os
import socket

import torch


def clear_gpu(synchronize=False):
    if synchronize and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        for step in (torch.cuda.empty_cache, torch.cuda.ipc_collect):
            try:
                step()
            except Exception:
                pass


def load_hf_model(loader, model_name, **kwargs):
    offline = os.environ.get('HF_HUB_OFFLINE', '').lower() in {'1', 'true', 'yes', 'on'}
    if not offline:
        try:
            socket.create_connection(('huggingface.co', 443), timeout=1).close()
        except OSError:
            offline = True
    kwargs.setdefault('local_files_only', offline)
    return loader.from_pretrained(model_name, **kwargs)
