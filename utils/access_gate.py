import json
import os
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from dotenv import load_dotenv

_MAGIC = b'DVAI'
_VERSION = 1
_SALT_SIZE = 16
_NONCE_SIZE = 12


def _make_key(secret, salt):
    kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1)
    return kdf.derive(secret.encode('utf-8'))


def open_payload(file_path):
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    secret = os.getenv('ENC')
    if not secret:
        raise ValueError('Invalid payload access.')
    input_path = Path(file_path)
    if not input_path.is_file():
        raise FileNotFoundError(f'File not found: {input_path}')
    with input_path.open('rb') as file:
        magic = file.read(4)
        version = file.read(1)
        salt = file.read(_SALT_SIZE)
        nonce = file.read(_NONCE_SIZE)
        ciphertext = file.read()
    if magic != _MAGIC:
        raise ValueError('Invalid payload file.')
    if not version or version[0] != _VERSION:
        raise ValueError('Unsupported payload file version.')
    if len(salt) != _SALT_SIZE:
        raise ValueError('Invalid salt data.')
    if len(nonce) != _NONCE_SIZE:
        raise ValueError('Invalid nonce data.')
    if not ciphertext:
        raise ValueError('Encrypted data is empty.')
    try:
        plaintext = AESGCM(_make_key(secret, salt)).decrypt(nonce, ciphertext, None)
    except InvalidTag as error:
        raise ValueError('Incorrect secret or corrupted payload file.') from error
    return json.loads(plaintext.decode('utf-8'))
