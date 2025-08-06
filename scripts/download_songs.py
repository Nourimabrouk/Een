#!/usr/bin/env python
import os
import yt_dlp

# --- Configuration ---
SONGS = {
    "U2 - One": "https://www.youtube.com/watch?v=ftjEcrrf7r0",
    "Bon Jovi - Always": "https://www.youtube.com/watch?v=9BMwcO6_hyA",
    "Foreigner - I Want to Know What Love Is": "https://www.youtube.com/watch?v=r3Pr1_v7hsw",
}
DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "website", "audio")


# --- Main Download Logic ---
def download_songs():
    """Downloads all songs using yt-dlp."""
    # Ensure the download directory exists
    if not os.path.exists(DOWNLOAD_DIR):
        print(f"Creating download directory: '{DOWNLOAD_DIR}'")
        os.makedirs(DOWNLOAD_DIR)

    for title, url in SONGS.items():
        print(f"Downloading '{title}'...")
        try:
            # Sanitize title for filename
            safe_title = "".join(c for c in title if c.isalnum() or c in " -_").rstrip()

            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": os.path.join(DOWNLOAD_DIR, f"{safe_title}.%(ext)s"),
                "quiet": True,
                "overwrites": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print(f"Successfully downloaded '{title}'")
        except Exception as e:
            print(f"Error downloading '{title}': {e}")


def main():
    """Main function to run the download process."""
    download_songs()
    print("\nAll song downloads complete!")


if __name__ == "__main__":
    main()
