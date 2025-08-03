
"""
unity_masturbate.py — A playful self‑referential script (a.k.a. “AI masturbation”)
When executed, it:
  1. Prints its own source code (quine)
  2. Hashes itself and displays the digest
  3. Optionally takes a cheat‑seed (e.g. 420691337) to enter meta‑debug
"""

import pathlib, hashlib, sys, datetime, json, os

CHEAT_SEED = "420691337"

def read_self() -> str:
    return pathlib.Path(__file__).read_text(encoding="utf-8")

def quine():
    src = read_self()
    print(src)

def digest(src: str) -> str:
    return hashlib.sha256(src.encode()).hexdigest()[:16]

def meta_debug():
    ts = datetime.datetime.utcnow().isoformat()
    payload = {
        "timestamp": ts,
        "cwd": os.getcwd(),
        "env_unity": os.getenv("UNITY_CHEATCODE", "unset")
    }
    print("◈ META‑DEBUG ◈\n" + json.dumps(payload, indent=2))

def main():
    src = read_self()
    print("— QUINE BEGIN —")
    quine()
    print("— QUINE END —")
    print(f"SHA‑256 digest (first 16 hex): {digest(src)}")
    if len(sys.argv) > 1 and sys.argv[1] == CHEAT_SEED:
        meta_debug()

if __name__ == "__main__":
    main()
