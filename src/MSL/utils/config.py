import configparser
import json
import re
import shutil
from pathlib import Path


_INTEGER = re.compile(r"^[+-]?\d+$")
_FLOAT = re.compile(
    r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$"
)


def _parse_value(raw_value: str):
    value = raw_value.strip()
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    if value.startswith(("[", "{", '"')):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON-style config value: {value!r}") from exc
    if _INTEGER.fullmatch(value):
        return int(value)
    if _FLOAT.fullmatch(value):
        return float(value)
    return value


def _section_target(cfg: dict, section: str) -> dict:
    if section == "config":
        return cfg
    target = cfg
    for part in section.split("."):
        if not part:
            raise ValueError(f"Invalid empty config section component: {section!r}")
        existing = target.get(part)
        if existing is None:
            existing = {}
            target[part] = existing
        if not isinstance(existing, dict):
            raise ValueError(f"Config section {section!r} conflicts with scalar key {part!r}.")
        target = existing
    return target


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path, _seen: set[Path] | None = None) -> dict:
    config_path = Path(path)
    if config_path.suffix.lower() != ".config":
        raise ValueError(f"Config file must use the .config extension: {config_path}")
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    seen = set() if _seen is None else set(_seen)
    if config_path in seen:
        chain = " -> ".join(str(item) for item in [*seen, config_path])
        raise ValueError(f"Circular config extends chain: {chain}")
    seen.add(config_path)

    parser = configparser.ConfigParser(
        interpolation=None,
        delimiters=("=",),
        comment_prefixes=("#", ";"),
        inline_comment_prefixes=None,
        strict=True,
        empty_lines_in_values=False,
    )
    parser.optionxform = str
    with config_path.open("r", encoding="utf-8-sig") as handle:
        parser.read_file(handle)
    if "config" not in parser:
        raise ValueError(f"Config file must contain a [config] section: {config_path}")

    cfg = {}
    for section in parser.sections():
        target = _section_target(cfg, section)
        for key, raw_value in parser.items(section, raw=True):
            clean_key = key.strip()
            if not clean_key:
                raise ValueError(f"Config section {section!r} contains an empty key.")
            target[clean_key] = _parse_value(raw_value)
    extends = cfg.pop("extends", None)
    if extends is None:
        return cfg
    parent_path = Path(str(extends))
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    parent = load_config(parent_path, _seen=seen)
    return _deep_merge(parent, cfg)


def _format_value(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    text = str(value)
    if "\n" in text or "\r" in text:
        return json.dumps(text, ensure_ascii=False)
    return text


def _flatten_sections(mapping: dict, prefix: str = ""):
    scalars = {}
    nested = []
    for key, value in mapping.items():
        if isinstance(value, dict):
            section = f"{prefix}.{key}" if prefix else str(key)
            nested.extend(_flatten_sections(value, section))
        else:
            scalars[str(key)] = value
    if prefix:
        return [(prefix, scalars), *nested]
    return [("config", scalars), *nested]


def write_config(cfg: dict, path: str | Path) -> Path:
    output_path = Path(path)
    if output_path.suffix.lower() != ".config":
        raise ValueError(f"Config snapshot must use the .config extension: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for section, values in _flatten_sections(cfg):
        if not values and section != "config":
            continue
        if lines:
            lines.append("")
        lines.append(f"[{section}]")
        lines.extend(f"{key}={_format_value(value)}" for key, value in values.items())
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def save_config_artifacts(source_path: str | Path, resolved_cfg: dict, output_dir: str | Path) -> dict:
    source = Path(source_path).resolve()
    if source.suffix.lower() != ".config" or not source.is_file():
        raise ValueError(f"Source config must be an existing .config file: {source}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    source_copy = destination / "source_config.config"
    resolved_snapshot = destination / "resolved_config.config"
    shutil.copy2(source, source_copy)
    write_config(resolved_cfg, resolved_snapshot)
    return {
        "source_config": str(source_copy),
        "resolved_config": str(resolved_snapshot),
    }
