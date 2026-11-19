"""Téléchargement du jeu NASA officiel (non redistribué dans ce projet).

Le ZIP NASA 5 contient éventuellement des ZIP imbriqués. On ne récupère
que B0005, B0006, B0007 et B0018. Aucun miroir non vérifié.
"""
import hashlib
import tempfile
import json
from pathlib import Path
import shutil
import urllib.request
from zipfile import ZipFile, BadZipFile
from .config import BATTERIES

NASA_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
MAX_MAT_BYTES = 100 * 1024 * 1024
MAX_NESTED_ZIP_BYTES = 800 * 1024 * 1024


def _copy_selected(archive, raw_dir, hashes):
    for member in archive.infolist():
        name = Path(member.filename.replace("\\", "/")).name.upper()
        if member.is_dir():
            continue
        if name in {battery + ".MAT" for battery in BATTERIES}:
            if member.file_size > MAX_MAT_BYTES:
                raise ValueError("Fichier MAT de taille excessive: " + name)
            target = Path(raw_dir) / (Path(name).stem + ".mat")
            digest = hashlib.sha256()
            with archive.open(member) as src, target.open("wb") as out:
                while True:
                    block = src.read(1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
                    out.write(block)
            hashes[target.stem] = digest.hexdigest()
        elif name.endswith(".ZIP"):
            if member.file_size > MAX_NESTED_ZIP_BYTES:
                raise ValueError("ZIP interne trop volumineux")
            # Temporaire sur disque : aucun chargement de plusieurs centaines
            # de Mo en RAM sur une machine Windows ordinaire.
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_zip = Path(temp_dir) / "nasa_nested.zip"
                with archive.open(member) as source, temp_zip.open("wb") as out:
                    shutil.copyfileobj(source, out, length=1024 * 1024)
                with ZipFile(temp_zip) as nested:
                    _copy_selected(nested, raw_dir, hashes)


def download_nasa(raw_dir, archive_path=None):
    """archive_path permet l'installation strictement hors-ligne."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = Path(archive_path) if archive_path else raw_dir / "nasa_original.zip"
    if not archive.exists():
        print("Telechargement officiel NASA :", NASA_URL)
        try:
            with urllib.request.urlopen(NASA_URL, timeout=120) as response, \
                    archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        except Exception:
            archive.unlink(missing_ok=True)
            raise
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    hashes = {}
    try:
        with ZipFile(archive) as z:
            _copy_selected(z, raw_dir, hashes)
    except (BadZipFile, OSError) as exc:
        raise RuntimeError("Archive NASA invalide: %s" % exc) from exc
    manifest = {"source": NASA_URL, "archive": str(archive),
                "archive_sha256": sha, "mat_sha256": hashes,
                "missing": [b for b in BATTERIES if b not in hashes]}
    (raw_dir / "provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not hashes:
        raise ValueError("Aucun fichier B0005/B0006/B0007/B0018 trouve dans l'archive")
    return manifest
