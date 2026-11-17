import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = json.loads((ROOT / "third_party/sources.json").read_text())
    for source in sources:
        destination = ROOT / "third_party" / source["directory"]
        patch = ROOT / source["patch"] if source.get("patch") else None
        identity = {"commit": source["commit"], "sha256": source["sha256"],
                    "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest() if patch else None}
        marker = destination / ".teleop-source.json"
        if destination.exists():
            if marker.is_file() and json.loads(marker.read_text()) == identity:
                print(f"Already installed: {source['directory']} @ {source['commit']}")
                continue
            raise RuntimeError(f"Refusing to overwrite {destination}; move the existing directory aside and retry")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".fetch-", dir=destination.parent) as temporary:
            staging = Path(temporary)
            archive = staging / "source.tar.gz"
            url = f"https://codeload.github.com/{source['repository']}/tar.gz/{source['commit']}"
            print(f"Downloading {url}", flush=True)
            with urlopen(url, timeout=60) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
            if hashlib.sha256(archive.read_bytes()).hexdigest() != source["sha256"]:
                raise RuntimeError(f"Checksum mismatch: {source['directory']}")
            unpacked = staging / "unpacked"
            with tarfile.open(archive) as bundle:
                bundle.extractall(unpacked, filter="data")
            roots = list(unpacked.iterdir())
            if len(roots) != 1 or not roots[0].is_dir():
                raise RuntimeError("Expected one archive root directory")
            source_root = roots[0]
            if patch:
                subprocess.run(["patch", "--batch", "--forward", "-p1", "-i", str(patch)],
                               cwd=source_root, check=True)
            (source_root / ".teleop-source.json").write_text(json.dumps(identity, indent=2) + "\n")
            source_root.rename(destination)
            print(f"Installed {destination}")


if __name__ == "__main__":
    main()
