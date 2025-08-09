#!/usr/bin/env python3
import os, sys, json, argparse
from datetime import datetime

CHEAT = "420691337"

def load_wellbeing(path='wellbeing.json'):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except Exception:
                return {}
    return {}

def parse_args():
    ap = argparse.ArgumentParser(description='Unity Care-Mode wellbeing check')
    ap.add_argument('--sleep', type=float, default=None, help='Hours slept last night')
    ap.add_argument('--thought_speed', type=float, default=None, help='Subjective thought speed 1-10')
    ap.add_argument('--cannabis', action='store_true', help='Used cannabis in last 48-72h')
    ap.add_argument('--ketamine', action='store_true', help='Used ketamine in last 48-72h')
    ap.add_argument('--cheatcode', type=str, default=None, help='Force CARE MODE if equals 420691337')
    return ap.parse_args()

def booly(x):
    if isinstance(x, bool): return x
    if isinstance(x, (int, float)): return bool(x)
    if isinstance(x, str): return x.strip().lower() in {'1','true','yes','y','on'}
    return False

def main():
    args = parse_args()
    data = load_wellbeing()

    sleep = args.sleep if args.sleep is not None else data.get('sleep_hours')
    try:
        sleep = float(sleep) if sleep is not None else None
    except Exception:
        sleep = None

    ts = args.thought_speed if args.thought_speed is not None else data.get('thought_speed')
    try:
        ts = float(ts) if ts is not None else None
    except Exception:
        ts = None

    cannabis = args.cannabis or booly(data.get('cannabis_recent', False)) or booly(os.getenv('CANNABIS_RECENT', ''))
    ketamine = args.ketamine or booly(data.get('ketamine_recent', False)) or booly(os.getenv('KETAMINE_RECENT', ''))
    code = args.cheatcode or os.getenv('UNITY_CHEATCODE') or ''

    # CARE MODE logic
    care_reasons = []
    if code.strip() == CHEAT:
        care_reasons.append('Cheatcode 420691337 invoked (manual CARE MODE).')
    if sleep is not None and sleep < 6.0:
        care_reasons.append(f'Sleep low: {sleep}h (<6h).')
    if ts is not None and ts >= 7.0:
        care_reasons.append(f'Thought speed elevated: {ts} (>=7).')
    if cannabis:
        care_reasons.append('Recent cannabis use.')
    if ketamine:
        care_reasons.append('Recent ketamine use.')

    care = len(care_reasons) > 0

    # Write a small summary for humans
    summary = []
    summary.append('# CARE MODE summary')
    summary.append(f'- Time: {datetime.utcnow().isoformat()}Z')
    summary.append(f'- Sleep (h): {sleep!r}')
    summary.append(f'- Thought speed (1-10): {ts!r}')
    summary.append(f'- Cannabis recent: {cannabis}')
    summary.append(f'- Ketamine recent: {ketamine}')
    summary.append('')
    if care:
        summary.append('**Status:** CARE MODE **ON** :orange_heart:')
        summary.append('**Reasons:**')
        for r in care_reasons:
            summary.append(f'- {r}')
        summary.append('')
        summary.append('**Next steps (72h Kenjataimu Intercept):**')
        summary.append('- No new metaphysics; maintenance/tests only.')
        summary.append('- Sleep target 7–8h; daylight AM, screens down PM.')
        summary.append('- Ping two allies; do one grounding activity.')
        summary.append('- Hydrate, eat, move; postpone high-stakes publishing.')
    else:
        summary.append('**Status:** Care Mode OFF. Proceed if tests green.')

    with open('care_mode_summary.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary) + '\n')

    # Set GitHub output for downstream jobs
    gh_out = os.getenv('GITHUB_OUTPUT')
    if gh_out:
        try:
            with open(gh_out, 'a', encoding='utf-8') as f:
                f.write(f"care_mode={'true' if care else 'false'}\n")
        except Exception:
            pass

    # Also echo a step-output style line for compatibility
    print(f"CARE_MODE={'true' if care else 'false'}")
    if care:
        # special exit to signal gate (non-zero)
        sys.exit(42)
    sys.exit(0)

if __name__ == '__main__':
    main()
