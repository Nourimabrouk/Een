# Unity Surprise Kit

Generated: 2025-08-09T21:47:51.462678Z

This kit contains:
1. **omega_care_mode.yml** — a reusable/dispatchable GitHub Action that gates high-novelty workflows behind a WELLBEING check and your cheatcode **420691337** (forces CARE MODE).
2. **check_wellbeing.py** — reads `wellbeing.json` (or CLI/env) and sets CARE MODE if any of:
   - sleep < 6h
   - thought_speed ≥ 7
   - recent cannabis or ketamine
   - cheatcode 420691337
   It writes `care_mode_summary.md` and exits 42 when CARE MODE is active.
3. **unity_lab.py** — tiny executable demo of contexts where `1+1=1` holds by design
   (boolean OR, set union, tropical max) plus a φ-convergence toy. Saves `unity_lab_convergence.png`.

## How to use
- Add `omega_care_mode.yml` to `.github/workflows/` in your repo.
- Place `check_wellbeing.py` into `scripts/` and create an optional `wellbeing.json`, e.g.:
  ```json
  {"sleep_hours": 5.0, "thought_speed": 8, "cannabis_recent": true, "ketamine_recent": false}
  ```
- Call the workflow via `workflow_dispatch` (you can pass `cheatcode=420691337`), or let it run on `push` to `wellbeing.json`.
- Gate your heavy jobs by making them depend on the `check` job and adding `if: needs.check.outputs.care_mode != 'true'`.

## Unity Lab quick run
```
python unity_lab.py
```
This prints examples and writes `unity_lab_convergence.png`.

— With love and rigor, your MetaBro Surprise Toolkit.
