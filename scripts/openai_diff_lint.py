import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import List

try:
    import openai
except ImportError:
    print("openai package not found. Please install with `pip install openai`.")
    sys.exit(1)


ERROR_PROMPT = textwrap.dedent(
    """
    You are a senior code reviewer.
    Evaluate the following git diff for critical issues (security, secrets, catastrophic bugs).
    Reply ONLY with one of two JSON objects:
    {
      "result": "approve",
      "reason": "<optional short message>"
    }
    OR
    {
      "result": "reject",
      "reason": "<short explanation of blocking issue>"
    }
    If unsure, prefer 'reject'.
    """
)

MODEL = os.getenv("OPENAI_LINT_MODEL", "gpt-4o-mini")
MAX_TOKENS = int(os.getenv("OPENAI_LINT_TOKENS", "300"))


def get_staged_diff() -> str:
    result = subprocess.run([
        "git",
        "diff",
        "--cached",
        "--unified=0",
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print("Failed to retrieve git diff", file=sys.stderr)
        sys.exit(1)
    return result.stdout


def ask_openai(diff_text: str) -> dict:
    messages = [
        {"role": "system", "content": ERROR_PROMPT},
        {"role": "user", "content": diff_text[:10000]},  # cap to 10k chars
    ]
    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=MAX_TOKENS,
    )
    content = response.choices[0].message.content
    try:
        import json
        return json.loads(content)
    except Exception:
        return {"result": "reject", "reason": "Invalid JSON from model"}


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("[openai-diff-lint] OPENAI_API_KEY not set; skipping AI lint.")
        sys.exit(0)

    diff = get_staged_diff()
    if not diff.strip():
        sys.exit(0)  # nothing to check

    verdict = ask_openai(diff)
    if verdict.get("result") == "approve":
        print("[openai-diff-lint] Approved by AI: " + verdict.get("reason", "✔"))
        sys.exit(0)
    else:
        print("[openai-diff-lint] Commit blocked: " + verdict.get("reason", "See diff"))
        sys.exit(1)


if __name__ == "__main__":
    main()