"""Package a prepared dataset for upload to Drive with a checksum."""

import hashlib
import zipfile
from pathlib import Path


def package_dataset(source: Path, archive: Path) -> str:
    source = source.resolve()
    if not (source / "dataset_card.json").is_file():
        raise ValueError("Source is not an exported dataset")
    if archive.exists():
        raise FileExistsError(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        for file in sorted(source.rglob("*")):
            if file.is_file():
                bundle.write(file, arcname=file.relative_to(source).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
        if bad:
            raise ValueError(f"Archive CRC failed: {bad}")
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    checksum = digest.hexdigest()
    archive.with_name(archive.name + ".sha256").write_text(
        f"{checksum}  {archive.name}\n", encoding="ascii"
    )
    return checksum
