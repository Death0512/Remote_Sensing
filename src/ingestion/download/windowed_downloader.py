import hashlib
import os
from pathlib import Path
from typing import Iterator, Union
import requests

from src.ingestion.exceptions import DataCorruptedError, DataUnavailableError


class WindowedDownloader:
    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        chunk_size: int = 65536,
        session: requests.Session | None = None,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.chunk_size = chunk_size
        self._session = session or requests.Session()

    @staticmethod
    def compute_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
        path = Path(file_path)
        if not path.is_file():
            raise DataUnavailableError(f"File not found: {path}")
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest().lower()

    @staticmethod
    def verify_checksum(file_path: Union[str, Path], expected_sha256: str) -> bool:
        actual = WindowedDownloader.compute_sha256(file_path)
        return actual == expected_sha256.strip().lower()

    def _resolve_local_path(self, uri: str) -> Path:
        clean = uri.removeprefix("file://")
        return Path(clean)

    def download_window(
        self,
        url_or_path: Union[str, Path],
        start_byte: int,
        end_byte: int,
        output_path: Union[str, Path, None] = None,
        expected_sha256: str | None = None,
    ) -> bytes | Path:
        if start_byte < 0 or end_byte < start_byte:
            raise ValueError(f"Invalid byte range: [{start_byte}, {end_byte}]")

        expected_len = end_byte - start_byte + 1
        uri_str = str(url_or_path)

        if uri_str.startswith("http://") or uri_str.startswith("https://"):
            data = self._fetch_http_window(uri_str, start_byte, end_byte, expected_len)
        else:
            data = self._fetch_local_window(self._resolve_local_path(uri_str), start_byte, expected_len)

        if len(data) != expected_len:
            raise DataCorruptedError(
                f"Window length mismatch: expected {expected_len} bytes, received {len(data)}"
            )

        if expected_sha256:
            actual_sha = hashlib.sha256(data).hexdigest().lower()
            if actual_sha != expected_sha256.strip().lower():
                raise DataCorruptedError(
                    f"SHA-256 mismatch for window: expected {expected_sha256}, got {actual_sha}"
                )

        if output_path is not None:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            part_file = out_file.with_name(f"{out_file.name}.part")
            try:
                part_file.write_bytes(data)
                part_file.replace(out_file)
            except Exception as exc:
                if part_file.exists():
                    part_file.unlink()
                raise exc
            return out_file

        return data

    def _fetch_local_window(self, path: Path, start_byte: int, length: int) -> bytes:
        if not path.is_file():
            raise DataUnavailableError(f"Local file unavailable: {path}")
        file_size = path.stat().st_size
        if start_byte >= file_size:
            raise DataUnavailableError(
                f"Requested start byte {start_byte} exceeds file size {file_size}"
            )
        with open(path, "rb") as f:
            f.seek(start_byte)
            return f.read(length)

    def _fetch_http_window(
        self,
        url: str,
        start_byte: int,
        end_byte: int,
        expected_len: int,
    ) -> bytes:
        headers = {"Range": f"bytes={start_byte}-{end_byte}"}
        try:
            resp = self._session.get(url, headers=headers, timeout=self.timeout)
            if resp.status_code in (404, 410):
                raise DataUnavailableError(f"Resource unavailable (HTTP {resp.status_code}): {url}")
            if resp.status_code in (500, 502, 503, 504):
                raise DataUnavailableError(f"Server error (HTTP {resp.status_code}): {url}")
            resp.raise_for_status()

            if resp.status_code == 206:
                return resp.content[:expected_len]
            elif resp.status_code == 200:
                return resp.content[start_byte : end_byte + 1]
            else:
                return resp.content[:expected_len]
        except (DataUnavailableError, DataCorruptedError):
            raise
        except requests.exceptions.RequestException as exc:
            raise DataUnavailableError(f"HTTP window request failed for {url}: {exc}") from exc

    def download_file(
        self,
        url_or_path: Union[str, Path],
        output_path: Union[str, Path],
        expected_sha256: str | None = None,
        chunk_size: int | None = None,
    ) -> Path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        part_file = out_file.with_name(f"{out_file.name}.part")
        hasher = hashlib.sha256()
        c_size = chunk_size or self.chunk_size

        try:
            if part_file.exists():
                part_file.unlink()

            uri_str = str(url_or_path)
            if uri_str.startswith("http://") or uri_str.startswith("https://"):
                try:
                    with self._session.get(uri_str, stream=True, timeout=self.timeout) as resp:
                        if resp.status_code in (404, 410):
                            raise DataUnavailableError(f"Remote file not found (HTTP {resp.status_code}): {uri_str}")
                        if resp.status_code in (500, 502, 503, 504):
                            raise DataUnavailableError(f"Remote server error (HTTP {resp.status_code}): {uri_str}")
                        resp.raise_for_status()

                        with open(part_file, "wb") as f:
                            for chunk in resp.iter_content(chunk_size=c_size):
                                if chunk:
                                    f.write(chunk)
                                    hasher.update(chunk)
                except requests.exceptions.RequestException as exc:
                    raise DataUnavailableError(f"Download stream error for {uri_str}: {exc}") from exc
            else:
                local_path = self._resolve_local_path(uri_str)
                if not local_path.is_file():
                    raise DataUnavailableError(f"Local source file not found: {local_path}")
                with open(local_path, "rb") as src, open(part_file, "wb") as dst:
                    while chunk := src.read(c_size):
                        dst.write(chunk)
                        hasher.update(chunk)

            actual_sha = hasher.hexdigest().lower()
            if expected_sha256:
                if actual_sha != expected_sha256.strip().lower():
                    if part_file.exists():
                        part_file.unlink()
                    raise DataCorruptedError(
                        f"Checksum mismatch for {output_path}: expected {expected_sha256}, got {actual_sha}"
                    )

            part_file.replace(out_file)
            return out_file
        except Exception:
            if part_file.exists():
                part_file.unlink()
            raise
