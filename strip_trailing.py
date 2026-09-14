from pathlib import Path

# Change this if your data folder is somewhere else
data = Path(__file__).resolve().parent / "data"

for f in data.glob("*.tbl"):
    lines = [ln.rstrip("\n").rstrip("|") for ln in open(f, encoding="utf-8", errors="replace")]
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Cleaned:", f.name)

print("Done. You can now load the data into PostgreSQL.")