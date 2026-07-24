import os
import pickle
import hashlib
import inspect
from pathlib import Path

class CacheManager:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _hash_bytes(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def hash_dataset(self, df):
        """Hash basé sur les tirages."""
        raw = df.to_csv(index=False).encode("utf-8")
        return self._hash_bytes(raw)

    def hash_params(self, cfg):
        """Hash basé sur les paramètres du modèle."""
        raw = str(cfg.__dict__).encode("utf-8")
        return self._hash_bytes(raw)

    def hash_code(self, func):
        """Hash basé sur le code source des features."""
        raw = inspect.getsource(func).encode("utf-8")
        return self._hash_bytes(raw)

    def make_cache_key(self, dataset_hash, params_hash, code_hash, name):
        """Clé unique pour un cache."""
        key = f"{name}_{dataset_hash}_{params_hash}_{code_hash}"
        return self.cache_dir / f"{key}.pkl"

    def load(self, path):
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def save(self, path, obj):
        with open(path, "wb") as f:
            pickle.dump(obj, f)
