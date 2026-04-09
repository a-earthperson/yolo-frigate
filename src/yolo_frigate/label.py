from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

import yaml

logger = logging.getLogger(__name__)


def parse_classes(file: str) -> list[str]:
    if file is None:
        raise ValueError("File path cannot be None")

    logger.debug("Opening class vocabulary file: %s", file)
    try:
        with open(file, encoding="utf-8") as content:
            if file.endswith(".yaml") or file.endswith(".yml"):
                logger.debug("Processing class vocabulary as YAML.")
                classes = _parse_yaml_classes(yaml.safe_load(content))
            else:
                logger.debug("Processing class vocabulary as text.")
                classes = _parse_text_classes(content)
    except Exception:
        logger.exception("Failed to parse class vocabulary file: %s", file)
        raise

    if not classes:
        raise ValueError(f"Class vocabulary file '{file}' did not define any classes.")
    return classes


def _parse_yaml_classes(payload) -> list[str]:
    if isinstance(payload, Mapping) and "names" in payload:
        payload = payload["names"]

    if isinstance(payload, Mapping):
        items = payload.items()
        if all(_is_int_like(key) for key in payload):
            items = sorted(items, key=lambda item: int(item[0]))
        return [_normalize_class_name(name) for _, name in items]

    if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes)):
        return [_normalize_class_name(name) for name in payload]

    raise ValueError(
        "YAML class vocabulary must be a list of class names or a mapping under 'names'."
    )


def _parse_text_classes(lines: Iterable[str]) -> list[str]:
    classes: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split(maxsplit=1)
        if len(parts) == 2 and _is_int_like(parts[0]):
            classes.append(_normalize_class_name(parts[1]))
            continue
        classes.append(_normalize_class_name(line))
    return classes


def _normalize_class_name(value: object) -> str:
    name = str(value).strip()
    if not name:
        raise ValueError("Encountered an empty class name in the label map.")
    return name


def _is_int_like(value: object) -> bool:
    try:
        int(str(value))
    except ValueError:
        return False
    return True
