import json
import os
from pathlib import Path

from dotenv import load_dotenv

class ConfigManager:
    def __init__(self, base_dir, default_json):
        self.base_dir = Path(base_dir)
        self.default_json = Path(default_json)
        self.env_path = self.base_dir / '.env'

    @staticmethod
    def load_env(base_dir):
        load_dotenv(Path(base_dir) / '.env', override=True)

    @staticmethod
    def env_value(name, default):
        return str(os.getenv(name) or default).strip().strip('"').strip("'")

    @staticmethod
    def parse_json_list(value):
        text = str(value or '').strip().strip('"').strip("'")
        if not (text.startswith('[') and text.endswith(']')):
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = json.loads(text.replace('\\', '\\\\'))
            except json.JSONDecodeError:
                return None
        return parsed if isinstance(parsed, list) else None

    @staticmethod
    def parse_json_dict(value):
        text = str(value or '').strip().strip('"').strip("'")
        if not (text.startswith('{') and text.endswith('}')):
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = json.loads(text.replace('\\', '\\\\'))
            except json.JSONDecodeError:
                return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _parse_env_value(value):
        return value.strip().strip('"').strip("'")

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
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
            elif ':' in line:
                key, value = line.split(':', 1)
            else:
                continue
            values[key.strip()] = self._parse_env_value(value)
        return values

    def discover_models(self, model_dir):
        env_value = self.read_env_file().get('SD_INPAINT_MODEL', '').strip()
        paths = self.parse_json_list(env_value) or ([env_value] if env_value else [])
        if model_dir.exists():
            paths.extend(str(path) for path in sorted(model_dir.glob('*.safetensors'), key=lambda item: item.name.lower()))
        models = {}
        seen = set()
        for raw_path in paths:
            if not raw_path:
                continue
            path = Path(str(raw_path).strip())
            if not path.is_absolute():
                path = self.base_dir / path
            path = path.resolve(strict=False)
            key = str(path).lower()
            if key in seen or not path.exists() or path.suffix.lower() != '.safetensors':
                continue
            seen.add(key)
            models[path.stem] = str(path)
        return models, next(iter(models), '')

    def discover_loras(self):
        env_value = self.read_env_file().get('SD_15_LoRA_MODEL', '').strip()
        if not env_value:
            return []
        paths = self.parse_json_list(env_value) or [env_value]
        resolved = []
        seen = set()
        for raw_path in paths:
            if not raw_path:
                continue
            path = Path(str(raw_path).strip())
            if not path.is_absolute():
                path = self.base_dir / path
            path = path.resolve(strict=False)
            key = str(path).lower()
            if key not in seen and path.exists() and path.suffix.lower() in {'.safetensors', '.bin'}:
                seen.add(key)
                resolved.append(str(path))
        return resolved

    def discover_embeddings(self):
        embeddings = self.parse_json_dict(self.read_env_file().get('SD_15_EMBEDDING', '')) or {}
        resolved = {}
        for raw_path, token in embeddings.items():
            path = Path(str(raw_path).strip())
            if not path.is_absolute():
                path = self.base_dir / path
            path = path.resolve(strict=False)
            if path.exists() and path.suffix.lower() in {'.safetensors', '.pt', '.bin'}:
                resolved[str(path)] = str(token).strip()
        return resolved

    def read_runtime_settings(self, model_dir):
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

    def write_env_key(self, key, value):
        lines = self._read_env_lines()
        formatted = json.dumps(value) if isinstance(value, list) else str(value)
        updated = []
        written = False
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                updated.append(line)
                continue
            if '=' in stripped:
                current_key, _ = stripped.split('=', 1)
            elif ':' in stripped:
                current_key, _ = stripped.split(':', 1)
            else:
                updated.append(line)
                continue
            if current_key.strip() == key:
                updated.append(f'{key}={formatted}')
                written = True
            else:
                updated.append(line)
        if not written:
            updated.append(f'{key}={formatted}')
        self.env_path.write_text('\n'.join(updated) + '\n', encoding='utf-8')

    def write_env_settings(self, active_config_path, autosave_enabled):
        values = self.read_env_file()
        values['JSON_config'] = self.relative_config_path(active_config_path)
        values['JSON_autosave'] = 'True' if autosave_enabled else 'False'

        lines = []
        written = set()
        for line in self._read_env_lines():
            stripped = line.strip()
            if not stripped or stripped.startswith('#') or ':' not in stripped:
                if not stripped or stripped.startswith('#') or '=' not in stripped:
                    lines.append(line)
                    continue
                key = stripped.split('=', 1)[0].strip()
            else:
                key = stripped.split(':', 1)[0].strip()
            if key in {'JSON_config', 'JSON_autosave'}:
                if key == 'JSON_config':
                    lines.append(f'JSON_config="{values["JSON_config"]}"')
                else:
                    lines.append(f'JSON_autosave={values["JSON_autosave"]}')
                written.add(key)
            else:
                lines.append(line)

        for key, value in [('JSON_config', f'"{values["JSON_config"]}"'), ('JSON_autosave', values['JSON_autosave'])]:
            if key not in written:
                prefix = 'JSON_config' if key == 'JSON_config' else 'JSON_autosave'
                lines.append(f'{prefix}={value}')

        self.env_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    def reorder_model_list_in_env(self, selected_model_path):
        if not selected_model_path:
            return
        selected_path = str(Path(selected_model_path).resolve(strict=False))
        env_value = self.read_env_file().get('SD_INPAINT_MODEL', '').strip()
        current = self.parse_json_list(env_value) or ([env_value] if env_value else [])
        ordered = [selected_path]
        seen = {selected_path.lower()}
        for raw in current:
            candidate = str(Path(str(raw)).resolve(strict=False))
            key = candidate.lower()
            if key not in seen:
                ordered.append(candidate)
                seen.add(key)
        self.write_env_key('SD_INPAINT_MODEL', ordered)

    def relative_config_path(self, active_config_path):
        active_config_path = Path(active_config_path)
        try:
            return active_config_path.relative_to(self.base_dir).as_posix()
        except Exception:
            return str(active_config_path).replace('\\', '/')

    def relative_display_path(self, path):
        path = Path(path)
        try:
            return path.relative_to(self.base_dir).as_posix()
        except Exception:
            return str(path).replace('\\', '/')

    def refresh_config_files(self, base_dir):
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

    def _parse_and_save_json(self, active_config_path, text, autosave_enabled):
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError('JSON must be an object.')
        if autosave_enabled:
            Path(active_config_path).write_text(text, encoding='utf-8')
        return data

    def save_json(self, active_config_path, text, autosave_enabled):
        return self._parse_and_save_json(active_config_path, text, autosave_enabled)

    def sync_config(self, active_config_path, text, autosave_enabled):
        return self._parse_and_save_json(active_config_path, text, autosave_enabled)
