import json
from pathlib import Path


class ConfigManager:
    def __init__(self, base_dir, default_json):
        self.base_dir = Path(base_dir)
        self.default_json = Path(default_json)
        self.env_path = self.base_dir / '.env'

    def read_env_file(self):
        values = {}
        if not self.env_path.exists():
            return values
        try:
            for line in self.env_path.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                elif ':' in line:
                    key, value = line.split(':', 1)
                else:
                    continue
                values[key.strip()] = value.strip().strip('"').strip("'")
        except Exception:
            return {}
        return values

    def discover_models(self, model_dir):
        env_value = self.read_env_file().get('SD_INPAINT_MODEL', '').strip()
        paths = []
        if env_value:
            try:
                parsed = json.loads(env_value)
                if isinstance(parsed, list):
                    paths.extend(parsed)
                elif isinstance(parsed, str):
                    paths.append(parsed)
            except json.JSONDecodeError:
                paths.append(env_value)
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

    def write_env_settings(self, active_config_path, autosave_enabled):
        values = {}
        if self.env_path.exists():
            try:
                for line in self.env_path.read_text(encoding='utf-8').splitlines():
                    line = line.strip()
                    if not line or line.startswith('#') or ':' not in line:
                        continue
                    key, value = line.split(':', 1)
                    values[key.strip()] = value.strip().strip('"').strip("'")
            except Exception:
                pass
        values['JSON_config'] = self.relative_config_path(active_config_path)
        values['JSON_autosave'] = 'True' if autosave_enabled else 'False'
        lines = []
        written = set()
        if self.env_path.exists():
            try:
                for line in self.env_path.read_text(encoding='utf-8').splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith('#') or ':' not in stripped:
                        lines.append(line)
                        continue
                    key = stripped.split(':', 1)[0].strip()
                    if key == 'JSON_config':
                        lines.append(f'JSON_config: "{values["JSON_config"]}"')
                        written.add(key)
                    elif key == 'JSON_autosave':
                        lines.append(f'JSON_autosave: {values["JSON_autosave"]}')
                        written.add(key)
                    else:
                        lines.append(line)
            except Exception:
                lines = []
        if 'JSON_config' not in written:
            lines.append(f'JSON_config: "{values["JSON_config"]}"')
        if 'JSON_autosave' not in written:
            lines.append(f'JSON_autosave: {values["JSON_autosave"]}')
        self.env_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

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
        data_dir = Path(base_dir) / 'data' / 'diffusion_config'
        data_dir.mkdir(parents=True, exist_ok=True)
        return sorted([p for p in data_dir.glob('*.json') if p.is_file()], key=lambda p: p.name.lower())

    def load_config(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'Configuration file was not found:\n\n{path}')
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError(f'{path.name} must contain a JSON object.')
        return data, path

    def save_json(self, active_config_path, text, autosave_enabled):
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError('JSON must be an object.')
        if autosave_enabled:
            Path(active_config_path).write_text(text, encoding='utf-8')
        return data

    def sync_config(self, active_config_path, text, autosave_enabled):
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError('JSON must be an object.')
        if autosave_enabled:
            Path(active_config_path).write_text(text, encoding='utf-8')
        return data
