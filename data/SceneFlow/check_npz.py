"""Find unreadable .npz samples and (optionally) drop them from the split lists.

An interrupted np.savez_compressed leaves a 0-byte or truncated file behind.
np.load then raises `EOFError: No data left in file` inside a DataLoader worker,
which kills training mid-epoch. This script validates every entry listed in the
split txt files and can rewrite the lists without the bad ones.

Validation is cheap: it stats the file, then reads the zip central directory
(and, with --deep, the compressed member data as well).

Usage:
    python check_npz.py                       # report only
    python check_npz.py --deep                # also decompress every member (slow)
    python check_npz.py --prune               # rewrite the txt lists without bad entries
    python check_npz.py --prune --delete-bad  # also delete the broken npz files
"""

import argparse
import glob
import os
import shutil
import zipfile

PATTERN = "sceneflow_*_npz.txt"
EXPECTED_KEYS = ("left_clean", "right_clean", "left_final", "right_final", "disp")


def check_npz(path, deep=False, keys=EXPECTED_KEYS):
    """Return None if the file is usable, else a short reason string."""
    if not os.path.isfile(path):
        return "missing"
    if os.path.getsize(path) == 0:
        return "0 bytes"
    try:
        with zipfile.ZipFile(path) as zf:
            names = {os.path.splitext(n)[0] for n in zf.namelist()}
            missing = [k for k in keys if k not in names]
            if missing:
                return "missing keys: {}".format(", ".join(missing))
            if deep:
                bad = zf.testzip()
                if bad is not None:
                    return "corrupt member: {}".format(bad)
    except zipfile.BadZipFile:
        return "not a valid npz (truncated?)"
    except Exception as exc:  # noqa: BLE001 - report whatever np.load would hit
        return "{}: {}".format(type(exc).__name__, exc)
    return None


def resolve(entry, root):
    return entry if os.path.isabs(entry) else os.path.join(root, entry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--data-dir", default=here, help="directory holding the split txt files")
    parser.add_argument("--root", default="/media/taein/PortableSSD/TaeinKim/LightGEV/SceneFlow",
                        help="dataset root, used to resolve relative entries")
    parser.add_argument("--deep", action="store_true",
                        help="decompress every member (catches mid-file corruption; slow)")
    parser.add_argument("--prune", action="store_true",
                        help="rewrite the txt lists without the bad entries (.bak kept)")
    parser.add_argument("--delete-bad", action="store_true",
                        help="with --prune, also delete the broken npz files")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, PATTERN)))
    if not files:
        raise SystemExit("no files matching {} in {}".format(PATTERN, args.data_dir))

    total_bad = 0
    for txt in files:
        with open(txt) as f:
            lines = [l.strip() for l in f if l.strip()]

        good, bad = [], []
        for line in lines:
            entry = line.split()[0]
            reason = check_npz(resolve(entry, args.root), deep=args.deep)
            (good if reason is None else bad).append((line, reason))

        print("{}: {} entries, {} bad".format(os.path.basename(txt), len(lines), len(bad)))
        for line, reason in bad:
            print("   BAD  {}  [{}]".format(line.split()[0], reason))
        total_bad += len(bad)

        if not args.prune or not bad:
            continue

        if args.delete_bad:
            for line, _ in bad:
                path = resolve(line.split()[0], args.root)
                if os.path.isfile(path):
                    os.remove(path)
                    print("   deleted {}".format(path))

        shutil.copy2(txt, txt + ".bak")
        with open(txt, "w") as f:
            f.write("\n".join(line for line, _ in good) + "\n")
        print("   pruned -> {} entries (backup: {}.bak)".format(len(good), os.path.basename(txt)))

    print("\n{} bad entries total".format(total_bad))


if __name__ == "__main__":
    main()
