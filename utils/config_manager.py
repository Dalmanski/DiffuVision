import json
from pathlib import Path

from dotenv import load_dotenv

class ConfigManager:
    def __init__(self, base_dir, default_json):
        self.base_dir = Path(base_dir)
        self.default_json = Path(default_json)
        self.env_path = self.base_dir / '.env'

    @staticmethod
    def load_env(base_dir):
        env_path = Path(base_dir) / '.env'
        if not env_path.exists() or not env_path.read_text(encoding='utf-8').strip():
            env_path.write_text(
                'CTk_mode=system\n'
                'CTk_theme=default.json\n'
                'JSON_config=config/sd/default.json\n'
                'JSON_autosave=True\n'
                'SD_INPAINT_MODEL=[]\n'
                'SD_15_LoRA_MODEL=[]\n'
                'SD_15_EMBEDDING={}\n'
                'ENC=\n',
                encoding='utf-8',
            )
        load_dotenv(env_path, override=True)

    @staticmethod
    def _strip_quotes(value):
        return str(value or '').strip().strip('"').strip("'")

    @classmethod
    def _parse_json(cls, value, brackets, expected_type):
        text = cls._strip_quotes(value)
        if not (text.startswith(brackets[0]) and text.endswith(brackets[1])):
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = json.loads(text.replace('\\', '\\\\'))
            except json.JSONDecodeError:
                return None
        return parsed if isinstance(parsed, expected_type) else None

    @classmethod
    def parse_json_list(cls, value):
        return cls._parse_json(value, '[]', list)

    @classmethod
    def parse_json_dict(cls, value):
        return cls._parse_json(value, '{}', dict)

    @staticmethod
    def _split_env_line(line):
        line = line.strip()
        if not line or line.startswith('#'):
            return None
        for separator in ('=', ':'):
            if separator in line:
                key, value = line.split(separator, 1)
                return key.strip(), value
        return None

    def _read_env_lines(self):
        if not self.env_path.exists():
            return []
        try:
            return self.env_path.read_text(encoding='utf-8').splitlines()
        except Exception:
            return []

    def read_env_file(self):
        values = {}
        for line in self._read_env_lines():
            parsed = self._split_env_line(line)
            if parsed:
                values[parsed[0]] = self._strip_quotes(parsed[1])
        return values

    def _env_list(self, key):
        env_value = self.read_env_file().get(key, '').strip()
        return self.parse_json_list(env_value) or ([env_value] if env_value else [])

    def _resolve(self, raw_path):
        path = Path(str(raw_path).strip())
        if not path.is_absolute():
            path = self.base_dir / path
        return path.resolve(strict=False)

    def _discover_files(self, raw_paths, suffixes):
        resolved = []
        seen = set()
        for raw_path in raw_paths:
            if not raw_path:
                continue
            path = self._resolve(raw_path)
            key = str(path).lower()
            if key not in seen and path.exists() and path.suffix.lower() in suffixes:
                seen.add(key)
                resolved.append(path)
        return resolved

    def discover_models(self, model_dir):
        paths = self._env_list('SD_INPAINT_MODEL')
        if model_dir.exists():
            paths.extend(str(path) for path in sorted(model_dir.glob('*.safetensors'), key=lambda item: item.name.lower()))
        models = {path.stem: str(path) for path in self._discover_files(paths, {'.safetensors'})}
        return models, next(iter(models), '')

    def discover_loras(self):
        return [str(path) for path in self._discover_files(self._env_list('SD_15_LoRA_MODEL'), {'.safetensors', '.bin'})]

    def discover_embeddings(self):
        embeddings = self.parse_json_dict(self.read_env_file().get('SD_15_EMBEDDING', '')) or {}
        resolved = {}
        for raw_path, token in embeddings.items():
            path = self._resolve(raw_path)
            if path.exists() and path.suffix.lower() in {'.safetensors', '.pt', '.bin'}:
                resolved[str(path)] = str(token).strip()
        return resolved

    def read_settings(self, model_dir):
        values = self.read_env_file()
        models, default_model = self.discover_models(model_dir)
        loras = self.discover_loras()
        embeddings = self.discover_embeddings()
        config_value = values.get('JSON_config', '').replace('\\', '/')
        if config_value:
            start_config = Path(config_value)
            if not start_config.is_absolute():
                start_config = self.base_dir / start_config
        else:
            start_config = self.default_json
        autosave_value = values.get('JSON_autosave')
        autosave = True if autosave_value is None else autosave_value.strip().lower() in ('1', 'true', 'yes', 'on')
        return models, default_model, loras, embeddings, start_config, autosave

    def _write_env_values(self, updates):
        lines = []
        written = set()
        for line in self._read_env_lines():
            parsed = self._split_env_line(line)
            key = parsed[0] if parsed else None
            if key in updates:
                lines.append(f'{key}={updates[key]}')
                written.add(key)
            else:
                lines.append(line)
        lines.extend(f'{key}={value}' for key, value in updates.items() if key not in written)
        self.env_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    def write_env_key(self, key, value):
        self._write_env_values({key: json.dumps(value) if isinstance(value, list) else str(value)})

    def write_env_settings(self, active_config_path, autosave_enabled):
        self._write_env_values({
            'JSON_config': f'"{self.relative_path(active_config_path)}"',
            'JSON_autosave': 'True' if autosave_enabled else 'False',
        })

    def reorder_models(self, selected_model_path):
        if not selected_model_path:
            return
        selected_path = str(Path(selected_model_path).resolve(strict=False))
        ordered = [selected_path]
        seen = {selected_path.lower()}
        for raw in self._env_list('SD_INPAINT_MODEL'):
            candidate = str(Path(str(raw)).resolve(strict=False))
            key = candidate.lower()
            if key not in seen:
                ordered.append(candidate)
                seen.add(key)
        self.write_env_key('SD_INPAINT_MODEL', ordered)

    def relative_path(self, path):
        path = Path(path)
        try:
            return path.relative_to(self.base_dir).as_posix()
        except Exception:
            return str(path).replace('\\', '/')

    def list_configs(self, base_dir):
        config_dir = Path(base_dir) / 'config' / 'sd'
        return sorted([p for p in config_dir.rglob('*.json') if p.is_file()], key=lambda p: p.as_posix().lower())

    def load_config(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'Configuration file was not found:\n\n{path}')
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError(f'{path.name} must contain a JSON object.')
        return data, path

    def sync_config(self, active_config_path, text, autosave_enabled):
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError('JSON must be an object.')
        if autosave_enabled:
            Path(active_config_path).write_text(text, encoding='utf-8')
        return data

    save_json = sync_config
