"""Crash-safe JSONL persistence keyed by sample, method, and configuration."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 2


def configuration_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


class ResultStore:
    def __init__(
        self,
        path: str | Path,
        *,
        config_hash: str,
        manifest: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.config_hash = config_hash
        self.manifest_path = self.path.with_name(
            f"{self.path.stem}.{config_hash}.manifest.json"
        )
        if manifest is not None:
            self._write_manifest(manifest)
        self._rows = self._load()
        self._successful = {
            (row.get("config_hash"), row.get("sample_id"), row.get("method"))
            for row in self._rows
            if row.get("status") == "ok"
        }

    @property
    def rows(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def pending(self, method: str, sample_ids: Iterable[str]) -> list[str]:
        return [
            sample_id
            for sample_id in sample_ids
            if (self.config_hash, sample_id, method) not in self._successful
        ]

    def append(self, row: dict[str, Any]) -> None:
        stored = {
            "schema_version": SCHEMA_VERSION,
            "config_hash": self.config_hash,
            "recorded_at": datetime.now(UTC).isoformat(),
            **row,
        }
        required = {"sample_id", "method", "status"}
        missing = required.difference(stored)
        if missing:
            raise ValueError(f"Result row missing required fields: {sorted(missing)}")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(stored, sort_keys=True, ensure_ascii=False) + "\n"
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        self._rows.append(stored)
        if stored["status"] == "ok":
            self._successful.add(
                (stored["config_hash"], stored["sample_id"], stored["method"])
            )

    def _write_manifest(self, configuration: dict[str, Any]) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "config_hash": self.config_hash,
            "configuration": configuration,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.manifest_path)

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        rows: list[dict[str, Any]] = []
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    break
                raise
            if not isinstance(row, dict):
                raise ValueError(f"Invalid result row at line {index + 1}")
            rows.append(row)
        return rows
