import zipfile, glob, json, os, sys
base = sys.argv[1]
for split in ("training", "validation", "test"):
    files = glob.glob(os.path.join(base, split, "**", "*.zip"), recursive=True)
    print(split, len(files), "recordings")
first = sorted(glob.glob(os.path.join(base, "training", "**", "*.zip"), recursive=True))[0]
z = zipfile.ZipFile(first)
print("")
print("file:", os.path.basename(first), "| size MB:", round(os.path.getsize(first) / 1e6, 2))
print("contents:", z.namelist())
for n in z.namelist():
    if n.endswith(".sc"):
        with z.open(n) as fh:
            total = 0
            for i, line in enumerate(fh):
                total += 1
                if i < 8:
                    print(line.decode(errors="replace").rstrip()[:220])
        print("system call lines in this recording:", total)
    if n.endswith(".json"):
        print(n, z.read(n).decode(errors="replace")[:1200])
ex = 0
tot = 0
for f in glob.glob(os.path.join(base, "test", "**", "*.zip"), recursive=True):
    zz = zipfile.ZipFile(f)
    for n in zz.namelist():
        if n.endswith(".json"):
            m = json.loads(zz.read(n))
            tot += 1
            if m.get("exploit"):
                ex += 1
            if tot == 1:
                print("")
                print("keys in a test json:", list(m.keys()))
print("test recordings:", tot, "| with exploit flag true:", ex)
