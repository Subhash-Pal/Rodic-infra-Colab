import hashlib
import json
import re

import pytest

import deploy_assets as da


def make_remote(tmp_path, contents):
    remote = tmp_path / "remote"
    remote.mkdir()
    files = {}
    for name, data in contents.items():
        (remote / name).write_bytes(data)
        files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    models = tmp_path / "models"
    models.mkdir()
    manifest = {"base_url": "file:///nonexistent", "files": files}
    (models / "MANIFEST.json").write_text(json.dumps(manifest))
    return remote, models


def test_real_manifest_pins_both_checkpoints():
    manifest = da.load_manifest()
    assert set(manifest["files"]) == {"best_codebrim.pt", "best_sdnet.pt"}
    for info in manifest["files"].values():
        assert re.fullmatch(r"[0-9a-f]{64}", info["sha256"]) and info["size"] > 1_000_000
    assert manifest["source_commit"] in manifest["base_url"]       # URL is pinned to a commit, not a branch


def test_downloads_missing_files_once_and_verifies(tmp_path):
    remote, models = make_remote(tmp_path, {"a.pt": b"alpha" * 1000, "b.pt": b"beta" * 1000})
    seen = []
    paths = da.ensure_models(models, base_url=remote.as_uri(), on_download=seen.append)
    assert sorted(seen) == ["a.pt", "b.pt"] and paths["a.pt"].read_bytes() == b"alpha" * 1000
    seen.clear()
    da.ensure_models(models, base_url=remote.as_uri(), on_download=seen.append)
    assert seen == []                                               # present and verified: no download
    assert not list(models.glob("*.part"))


def test_env_var_overrides_manifest_url(tmp_path, monkeypatch):
    remote, models = make_remote(tmp_path, {"a.pt": b"x" * 500})
    monkeypatch.setenv("RODIC_MODEL_BASE_URL", remote.as_uri())
    assert (da.ensure_models(models)["a.pt"]).exists()


def test_corrupted_download_is_rejected_and_not_kept(tmp_path):
    remote, models = make_remote(tmp_path, {"a.pt": b"good" * 500})
    (remote / "a.pt").write_bytes(b"evil" * 500)                    # same size, different content
    with pytest.raises(da.ModelAssetError):
        da.ensure_models(models, base_url=remote.as_uri())
    assert not (models / "a.pt").exists() and not list(models.glob("*.part"))


def test_existing_file_with_wrong_checksum_is_an_error(tmp_path):
    remote, models = make_remote(tmp_path, {"a.pt": b"good" * 500})
    (models / "a.pt").write_bytes(b"replaced weights")
    with pytest.raises(da.ModelAssetError, match="checksum"):
        da.ensure_models(models, base_url=remote.as_uri())


def test_unreachable_source_raises_a_clear_error(tmp_path):
    _, models = make_remote(tmp_path, {"a.pt": b"x" * 100})
    with pytest.raises(da.ModelAssetError, match="Could not download"):
        da.ensure_models(models, base_url=(tmp_path / "nowhere").as_uri())
