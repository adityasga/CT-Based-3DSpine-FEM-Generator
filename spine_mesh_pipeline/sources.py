"""Downloads for the open datasets: plain files, OSF storage folders, and single members of a large remote zip.

Every download goes to a `.part` file first, so an interrupted transfer is never mistaken for a finished one,
and existing files are reused, so re-running a selection does not fetch anything twice.
"""
import io
import json
import shutil
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path

CHUNK = 1 << 20


def _open(url, headers=None, attempts=5):
    request = urllib.request.Request(url, headers={"User-Agent": "spine-mesh-pipeline", **(headers or {})})
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(request, timeout=120)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code in (429, 500, 502, 503, 504)
            if not retryable or attempt == attempts - 1:
                raise
            time.sleep(5 * 2 ** attempt)


def get_json(url):
    with _open(url) as response:
        return json.load(response)


def download(url, dest):
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with _open(url) as response, part.open("wb") as f:
        shutil.copyfileobj(response, f, CHUNK)
    part.replace(dest)
    return dest


def osf_list(folder_url):
    """{name: {kind, size, download, folder}} for an OSF storage folder given by its API URL."""
    entries = {}
    url = folder_url + ("&" if "?" in folder_url else "?") + "page%5Bsize%5D=100"
    while url:
        page = get_json(url)
        for item in page["data"]:
            attributes = item["attributes"]
            entries[attributes["name"]] = {
                "kind": attributes["kind"], "size": attributes.get("size"),
                "download": item["links"].get("download"),
                "folder": item["relationships"].get("files", {}).get("links", {}).get("related", {}).get("href"),
            }
        url = page["links"].get("next")
    return entries


class _HttpRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file; only used to let zipfile read the central directory."""

    def __init__(self, archive):
        self.archive, self.pos = archive, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.archive.size + offset}[whence]
        return self.pos

    def readinto(self, buffer):
        n = min(len(buffer), self.archive.size - self.pos)
        if n <= 0:
            return 0
        data = b"".join(self.archive._stream(self.pos, self.pos + n))
        buffer[:len(data)] = data
        self.pos += len(data)
        return len(data)


class RemoteZip:
    """Read single members of a zip, local or remote, without fetching the whole archive.

    A remote zip is read with HTTP range requests. Its central directory is read once and cached as JSON
    (name -> header offset, compressed size, size, method, CRC), keeping only the names `keep` accepts.
    """

    def __init__(self, source, index_file, keep=lambda name: True):
        self.source = str(source)
        self.remote = self.source.startswith(("http://", "https://"))
        index_file = Path(index_file)
        if index_file.exists():
            self.members = json.loads(index_file.read_text())
            return
        print(f"Reading the directory of {self.source} (once; cached in {index_file.name})", flush=True)
        if self.remote:
            with _open(self.source, {"Range": "bytes=0-0"}) as response:
                self.size = int(response.headers["Content-Range"].rsplit("/", 1)[1])
            handle = io.BufferedReader(_HttpRangeFile(self), buffer_size=4 * CHUNK)
        else:
            handle = open(self.source, "rb")
        with handle, zipfile.ZipFile(handle) as zf:
            self.members = {i.filename: [i.header_offset, i.compress_size, i.file_size, i.compress_type, i.CRC]
                            for i in zf.infolist() if keep(i.filename)}
        index_file.parent.mkdir(parents=True, exist_ok=True)
        index_file.write_text(json.dumps(self.members))

    def _stream(self, start, stop):
        """Yield the bytes [start, stop) of the archive in chunks."""
        if self.remote:
            with _open(self.source, {"Range": f"bytes={start}-{stop - 1}"}) as response:
                if response.status != 206:
                    raise RuntimeError(f"{self.source} ignored the byte-range request (HTTP {response.status})")
                while chunk := response.read(CHUNK):
                    yield chunk
        else:
            with open(self.source, "rb") as f:
                f.seek(start)
                remaining = stop - start
                while remaining > 0 and (chunk := f.read(min(CHUNK, remaining))):
                    remaining -= len(chunk)
                    yield chunk

    def _member_chunks(self, name):
        offset, compressed, size, method, crc = self.members[name]
        if method not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise RuntimeError(f"{name}: unsupported zip compression method {method}")
        # One request covers the local header (name + extra field, assumed < 1 KB of extra) and the data.
        header_size = 30 + len(name.encode()) + 1024
        header, in_data = b"", False
        decoder = zlib.decompressobj(-15) if method == zipfile.ZIP_DEFLATED else None
        check, produced, remaining = 0, 0, compressed
        for chunk in self._stream(offset, offset + header_size + compressed):
            if not in_data:
                header += chunk
                if len(header) < 30:
                    continue
                if header[:4] != b"PK\x03\x04":
                    raise RuntimeError(f"{name}: no local file header at offset {offset}")
                name_len, extra_len = struct.unpack("<HH", header[26:30])
                if extra_len > 1024:
                    raise RuntimeError(f"{name}: local extra field larger than expected ({extra_len} bytes)")
                if len(header) < 30 + name_len + extra_len:
                    continue
                chunk, in_data = header[30 + name_len + extra_len:], True
            chunk = chunk[:remaining]
            remaining -= len(chunk)
            data = decoder.decompress(chunk) if decoder else chunk
            check, produced = zlib.crc32(data, check), produced + len(data)
            yield data
            if remaining <= 0:
                break
        if decoder:
            data = decoder.flush()
            check, produced = zlib.crc32(data, check), produced + len(data)
            yield data
        if produced != size or check != crc:
            raise RuntimeError(f"{name}: corrupt download (size {produced}/{size}, CRC mismatch={check != crc})")

    def read(self, name):
        return b"".join(self._member_chunks(name))

    def extract(self, name, dest):
        dest = Path(dest)
        if dest.exists():
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        with part.open("wb") as f:
            for data in self._member_chunks(name):
                f.write(data)
        part.replace(dest)
        return dest
