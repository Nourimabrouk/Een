import os
import sys

"""
Download requested audio tracks into website/audio using yt-dlp.
Usage: python scripts/e2e/download_audio_tracks.py
Requires: yt-dlp installed in the active environment.
"""

TRACKS = [
    # Provided existing webm files are already in repo; we keep them in playlist.
    {
        "url": "https://www.youtube.com/watch?v=n8X9_MgEdCg",  # TheFatRat - Unity
        "out": "TheFatRat - Unity.%(ext)s",
        "format": "bestaudio/best",
        "ext": "mp3",
    },
    {
        "url": "https://www.youtube.com/watch?v=vdB-8eLEW8g",  # Bob Marley - One Love (Official)
        "out": "Bob Marley - One Love.%(ext)s",
        "format": "bestaudio/best",
        "ext": "mp3",
    },
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run(cmd: str) -> int:
    print(f"$ {cmd}")
    return os.system(cmd)


def main() -> int:
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    audio_dir = os.path.join(repo_root, "website", "audio")
    ensure_dir(audio_dir)

    # Verify yt-dlp is available
    rc = run(
        "yt-dlp --version >NUL 2>&1"
        if os.name == "nt"
        else "yt-dlp --version >/dev/null 2>&1"
    )
    if rc != 0:
        print("yt-dlp not found. Please install with: pip install yt-dlp")
        return 1

    for track in TRACKS:
        out_template = os.path.join(audio_dir, track["out"]).replace("\\", "/")
        # Prefer mp3 output for broad compatibility
        cmd = (
            f"yt-dlp -f {track['format']} --extract-audio --audio-format {track['ext']} "
            f"-o \"{out_template}\" \"{track['url']}\""
        )
        rc = run(cmd)
        if rc != 0:
            print(f"Failed to download: {track['url']}")

    print("Done. Files saved to website/audio/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
