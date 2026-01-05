import os
import sys
from pathlib import Path

print(f"User Home: {Path.home()}")

# Standard locations
candidates = [
    Path.home() / ".cache" / "huggingface" / "hub",
    Path.home() / ".huggingface",
    Path(os.getenv("LOCALAPPDATA", "")) / "huggingface"
]

found = False
for p in candidates:
    if p.exists():
        print(f"FOUND CACHE AT: {p}")
        # List contents
        try:
            print("Contents:")
            for child in p.iterdir():
                print(f" - {child.name}")
        except:
            pass
        found = True

if not found:
    print("Could not find standard HuggingFace cache folders.")
    print("Checked:")
    for p in candidates:
        print(f" - {p}")
