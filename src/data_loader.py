"""Load the Hugging Face Arrow dataset with PyArrow (falls back to `datasets` if needed).

The file is an Arrow IPC *stream* written by `datasets`. Image columns are
struct<bytes, path>; the bytes are the actual encoded images. We memory-map the file so
image bytes are read lazily and never copied into the text/embedding pipeline."""
from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa

log = logging.getLogger(__name__)

IMAGE_COLUMN = "image"


def load_dataset_table(path: str | Path) -> pa.Table:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Set DATASET_PATH in .env or copy the .arrow file to data/source/."
        )
    src = pa.memory_map(str(path))  # kept alive by the table's buffers
    try:
        return pa.ipc.open_stream(src).read_all()
    except pa.ArrowInvalid:
        pass
    try:
        return pa.ipc.open_file(pa.memory_map(str(path))).read_all()
    except pa.ArrowInvalid:
        pass
    try:  # last resort
        from datasets import Dataset

        return Dataset.from_file(str(path)).data.table
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Could not read {path} as an Arrow stream/file or HF dataset: {exc}") from exc


def table_to_records(table: pa.Table, limit: int | None = None) -> list[dict]:
    """Convert to plain dicts, excluding the image column (images never enter text pipelines)."""
    cols = [c for c in table.column_names if c != IMAGE_COLUMN]
    sub = table.select(cols)
    if limit:
        sub = sub.slice(0, limit)
    return sub.to_pylist()


_MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF8", b"RIFF")


class ImageStore:
    """Lazy lookup of image bytes by product id. Returns None unless the bytes look like a real image."""

    def __init__(self, table: pa.Table | None = None, records: list[dict] | None = None):
        self._has_images = bool(table is not None and IMAGE_COLUMN in table.column_names)
        self._col = table.column(IMAGE_COLUMN) if self._has_images else None
        self._row = {pid: i for i, pid in enumerate(table.column("id").to_pylist())} if table is not None else {}
        self._urls = {r["id"]: r.get("imageUrl") for r in (records or []) if r.get("imageUrl")}

    def get_url(self, product_id: int) -> str | None:
        return self._urls.get(product_id)

    def get_bytes(self, product_id: int) -> bytes | None:
        if not self._has_images:
            return None
        i = self._row.get(product_id)
        if i is None:
            return None
        cell = self._col[i].as_py() or {}
        data = cell.get("bytes")
        if data and data.startswith(_MAGIC):
            return data
        return None
