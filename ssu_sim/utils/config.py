"""JSON and YAML config loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            data = _load_simple_yaml(text)
            if data is None:
                raise RuntimeError(
                    "YAML config support requires PyYAML for this file. "
                    "Install the project with: pip install ."
                ) from exc
        else:
            # Expanded, source-backed model profiles are parsed repeatedly by
            # catalog/runtime validation. Keep safe construction while using
            # PyYAML's equivalent C parser when the installation provides it.
            data = yaml.load(text, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    else:
        raise ValueError(f"unsupported config format: {config_path}")
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    return data


def _load_simple_yaml(text: str) -> Any:
    """Parse the small YAML subset used by the bundled configs.

    PyYAML remains the primary parser. This fallback exists so local smoke tests
    and CLI examples still work in minimal environments without optional
    dependencies installed.
    """

    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        content = raw_line.split("#", 1)[0].rstrip()
        if not content:
            continue
        indent = len(content) - len(content.lstrip(" "))
        lines.append((indent, content.lstrip(" ")))
    if not lines:
        return {}

    index = 0

    def parse_block(indent: int) -> Any:
        nonlocal index
        if index >= len(lines) or lines[index][0] < indent:
            return {}
        if lines[index][1].startswith("- "):
            values = []
            while index < len(lines) and lines[index][0] == indent:
                item = lines[index][1]
                if not item.startswith("- "):
                    raise ValueError("cannot mix YAML list and mapping items")
                value_text = item[2:].strip()
                index += 1
                if value_text:
                    values.append(_parse_simple_scalar(value_text))
                else:
                    values.append(parse_block(indent + 2))
            return values

        mapping: dict[str, Any] = {}
        while index < len(lines) and lines[index][0] == indent:
            item = lines[index][1]
            if item.startswith("- "):
                break
            key, separator, value_text = item.partition(":")
            if not separator or not key.strip():
                raise ValueError(f"unsupported YAML line: {item}")
            index += 1
            value_text = value_text.strip()
            if value_text:
                mapping[key.strip()] = _parse_simple_scalar(value_text)
            elif index < len(lines) and lines[index][0] > indent:
                mapping[key.strip()] = parse_block(lines[index][0])
            else:
                mapping[key.strip()] = {}
        return mapping

    data = parse_block(lines[0][0])
    if index != len(lines):
        raise ValueError("unsupported YAML indentation")
    return data


def _parse_simple_scalar(text: str) -> Any:
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_simple_scalar(item.strip()) for item in inner.split(",")]
    lowered = text.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text
