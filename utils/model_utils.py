import gc
import os
import socket

import torch

_token_notice_shown = False


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


def _token_rejected(error):
    while error is not None:
        text = str(error).lower()
        if 'token has expired' in text or '401' in text:
            return True
        error = error.__cause__ or error.__context__
    return False


def _print_token_notice():
    global _token_notice_shown
    if _token_notice_shown:
        return
    _token_notice_shown = True
    print()
    print('[HF] Hugging Face token is expired or invalid. Continuing without it...')
    print('[HF] The models used by this app are public, so a token is not required.')
    print('[HF] To remove the old token, run:')
    print('[HF]   hf auth logout   (or: huggingface-cli logout)')
    print('[HF] To renew the token instead:')
    print('[HF]   1. Open https://huggingface.co/settings/tokens')
    print('[HF]   2. Create a new token with "Read" access and copy it')
    print('[HF]   3. Run: hf auth login   (or: huggingface-cli login)')
    print('[HF]   4. Paste the token and press Enter')
    print()


def load_hf_model(loader, model_name, **kwargs):
    offline = os.environ.get('HF_HUB_OFFLINE', '').lower() in {'1', 'true', 'yes', 'on'}
    if not offline:
        try:
            socket.create_connection(('huggingface.co', 443), timeout=1).close()
        except OSError:
            offline = True
    kwargs.setdefault('local_files_only', offline)
    try:
        return loader.from_pretrained(model_name, **kwargs)
    except OSError as error:
        if kwargs['local_files_only']:
            raise
        if _token_rejected(error):
            _print_token_notice()
    try:
        return loader.from_pretrained(model_name, **{**kwargs, 'token': False})
    except OSError:
        return loader.from_pretrained(model_name, **{**kwargs, 'local_files_only': True})
