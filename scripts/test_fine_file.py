from pathlib import Path
import os, sys
def find_all_files(root: str | Path, extension: str) -> dict[str, Path]:
    """Recursively find all files with the given `extension` in `root`."""
    return dict(sorted((p.stem, p) for p in Path(root).rglob(f"*{extension}")))

test_path = "/mnt/ceph_rbd/data/vctk/hubert_feature/large_l9"
dict_all_path = find_all_files(test_path, ".pt")

for idx, (key, value) in enumerate(dict_all_path.items()):
    print(key, value)
    if idx >= 10:
        break