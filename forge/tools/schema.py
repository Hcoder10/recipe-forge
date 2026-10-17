"""Coerce LLM-sent tool arguments to a JSON schema (the subset our tools use).

LLMs send objects as JSON strings, numbers as strings, booleans as "true", single items where a list is
expected. We fix what is unambiguous and report the rest as one short line.
"""
from __future__ import annotations

import json
from typing import Any


class ArgError(ValueError):
    pass


_TRUE = {"true", "yes", "1", "on", "y"}
_FALSE = {"false", "no", "0", "off", "n", "none", "null", ""}


def coerce(value: Any, schema: dict, path: str = "") -> Any:
    t = schema.get("type")
    if isinstance(t, list):
        last = None
        for alt in t:
            try:
                return coerce(value, {**schema, "type": alt}, path)
            except ArgError as e:
                last = e
        raise last or ArgError(f"{path or 'value'}: no matching type")
    if "enum" in schema and t in (None, "string"):
        v = str(value).strip() if value is not None else value
        if v in schema["enum"]:
            return v
        low = {str(e).lower(): e for e in schema["enum"]}
        if isinstance(v, str) and v.lower() in low:
            return low[v.lower()]
        raise ArgError(f"{path or 'value'}: must be one of {', '.join(map(str, schema['enum']))}")
    if t == "object":
        if isinstance(value, str):
            s = value.strip()
            if s == "":
                value = {}
            else:
                try:
                    value = json.loads(s)
                except json.JSONDecodeError:
                    raise ArgError(f"{path or 'value'}: expected a JSON object")
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ArgError(f"{path or 'value'}: expected an object")
        props = schema.get("properties")
        if props is None:
            return value
        out = {}
        for k, v in value.items():
            if k in props:
                out[k] = coerce(v, props[k], f"{path}.{k}" if path else k)
            elif schema.get("additionalProperties", True) is not False:
                out[k] = v
            else:
                raise ArgError(f"unknown argument {k!r}; allowed: {', '.join(props)}")
        for k in schema.get("required", []):
            if k not in out or out[k] is None:
                raise ArgError(f"missing required argument {k!r}")
        for k, p in props.items():
            if k not in out and "default" in p:
                out[k] = p["default"]
        return out
    if t == "array":
        if isinstance(value, str):
            s = value.strip()
            if s.startswith("["):
                try:
                    value = json.loads(s)
                except json.JSONDecodeError:
                    raise ArgError(f"{path or 'value'}: expected a JSON array")
            else:
                value = [value]
        if value is None:
            value = []
        if not isinstance(value, list):
            value = [value]
        items = schema.get("items") or {}
        return [coerce(v, items, f"{path}[{i}]") for i, v in enumerate(value)]
    if t == "integer":
        if isinstance(value, bool):
            raise ArgError(f"{path or 'value'}: expected an integer")
        if isinstance(value, int):
            return value
        try:
            f = float(str(value).strip())
        except ValueError:
            raise ArgError(f"{path or 'value'}: expected an integer")
        if f != int(f):
            raise ArgError(f"{path or 'value'}: expected an integer, got {value}")
        return int(f)
    if t == "number":
        if isinstance(value, bool):
            raise ArgError(f"{path or 'value'}: expected a number")
        if isinstance(value, (int, float)):
            return value
        try:
            return float(str(value).strip().rstrip("%"))
        except ValueError:
            raise ArgError(f"{path or 'value'}: expected a number")
    if t == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        s = str(value).strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
        raise ArgError(f"{path or 'value'}: expected true or false")
    if t == "string":
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return str(value)
    return value
