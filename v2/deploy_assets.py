"""deploy_assets.py - fetch and verify the model checkpoints for deployment.

Streamlit Community Cloud clones the repo and runs the app; the two 45 MB checkpoints are not stored here.
``ensure_models()`` makes sure they exist in ``v2/hosted/models/`` and match the SHA-256 pinned in
``MANIFEST.json``. If they are already there (for example committed, or copied by hand) nothing is downloaded.

Source of the files, in order:
  1. ``RODIC_MODEL_BASE_URL`` environment variable (http(s):// or file:// folder URL),
  2. ``base_url`` in the manifest (pinned commit of the hosted POC repository).

CLI:
  python v2/deploy_assets.py --check
  python v2/deploy_assets.py --update-manifest --from-dir /path/to/models
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Optional

HOSTED_DIR = Path(__file__).resolve().parent / "hosted"
MODELS_DIR = HOSTED_DIR / "models"
MANIFEST_PATH = MODELS_DIR / "MANIFEST.json"


class ModelAssetError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _download(url: str, dest: Path, expected_sha: str, expected_size: Optional[int], retries: int = 2) -> None:
    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        tmp = None
        try:
            with urllib.request.urlopen(url, timeout=60) as response, tempfile.NamedTemporaryFile(
                dir=dest.parent, suffix=".part", delete=False
            ) as out:
                tmp = Path(out.name)
                shutil.copyfileobj(response, out, length=1024 * 1024)
            if expected_size is not None and tmp.stat().st_size != expected_size:
                raise ModelAssetError(f"size {tmp.stat().st_size} != expected {expected_size}")
            if sha256_file(tmp) != expected_sha:
                raise ModelAssetError("SHA-256 mismatch (file changed or download corrupted)")
            os.replace(tmp, dest)
            return
        except Exception as exc:  # retry on any network / integrity problem
            last = exc
            if tmp is not None and tmp.exists():
                tmp.unlink()
    raise ModelAssetError(f"Could not download {dest.name} from {url}: {last}")


def ensure_models(
    models_dir: Path = MODELS_DIR,
    manifest_path: Optional[Path] = None,
    base_url: Optional[str] = None,
    on_download: Optional[Callable[[str], None]] = None,
) -> Dict[str, Path]:
    """Return {file name: path}, downloading anything missing. Raises ModelAssetError on failure."""
    models_dir = Path(models_dir)
    manifest = load_manifest(manifest_path or (models_dir / "MANIFEST.json"))
    base = (base_url or os.environ.get("RODIC_MODEL_BASE_URL") or manifest["base_url"]).rstrip("/")
    models_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for name, info in manifest["files"].items():
        dest = models_dir / name
        if dest.exists():
            if sha256_file(dest) != info["sha256"]:
                raise ModelAssetError(
                    f"{dest} does not match the pinned checksum in MANIFEST.json. If you replaced the weights "
                    "on purpose, run: python v2/deploy_assets.py --update-manifest"
                )
        else:
            if on_download:
                on_download(name)
            _download(f"{base}/{name}", dest, info["sha256"], info.get("size"))
        paths[name] = dest
    return paths


def update_manifest(models_dir: Path = MODELS_DIR, from_dir: Optional[Path] = None) -> dict:
    """Recompute sizes and hashes from the weights in ``from_dir`` (default: ``models_dir``)."""
    source = Path(from_dir or models_dir)
    manifest = load_manifest()
    for name in manifest["files"]:
        path = source / name
        if not path.exists():
            raise ModelAssetError(f"{path} not found")
        manifest["files"][name] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="download if missing and verify checksums")
    parser.add_argument("--update-manifest", action="store_true", help="rewrite MANIFEST.json from local weights")
    parser.add_argument("--from-dir", default=None)
    args = parser.parse_args()
    if args.update_manifest:
        print(json.dumps(update_manifest(from_dir=args.from_dir), indent=2))
        return 0
    paths = ensure_models(on_download=lambda n: print(f"downloading {n} ..."))
    for name, path in paths.items():
        print(f"OK  {name}  {path.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
