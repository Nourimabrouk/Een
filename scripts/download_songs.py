#!/usr/bin/env python
import os
import yt_dlp

# --- Configuration ---
# Add songs here: "Artist - Title": "YouTube URL"
SONGS = {
    # Core unity-themed tracks
    "U2 - One": "https://www.youtube.com/watch?v=ftjEcrrf7r0",
    "Bon Jovi - Always": "https://www.youtube.com/watch?v=9BMwcO6_hyA",
    "Foreigner - I Want to Know What Love Is": "https://www.youtube.com/watch?v=r3Pr1_v7hsw",
    "Bob Marley - One Love": "https://www.youtube.com/watch?v=oFRbZJXjWIA",
    "TheFatRat - Unity": "https://www.youtube.com/watch?v=n8X9_MgEdCg",
    # Optional ambient/zen suggestions (uncomment to include; replace URLs if needed)
    # "Liquid Mind - Peace": "https://www.youtube.com/watch?v=G5o9n8o0XSk",
    # "Deuter - Temple of Silence": "https://www.youtube.com/watch?v=A3bt7rj0w8Y",
    # "Brian Eno - Ambient 1/1": "https://www.youtube.com/watch?v=tF2wml-0C3k",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(SCRIPT_DIR, "..", "website", "audio")


# --- Main Download Logic ---
def download_songs():
    """Downloads all songs using yt-dlp."""
    # Ensure the download directory exists
    if not os.path.exists(DOWNLOAD_DIR):
        print("Creating download directory: '" + DOWNLOAD_DIR + "'")
        os.makedirs(DOWNLOAD_DIR)

    for title, url in SONGS.items():
        print("Downloading '" + title + "'...")
        try:
            # Sanitize title for filename
            safe_title = "".join(c for c in title if c.isalnum() or c in " -_").rstrip()

            ydl_opts = {
                # Download best available audio format (skip conversion for now)
                "format": "bestaudio/best",
                "outtmpl": os.path.join(DOWNLOAD_DIR, safe_title + ".%(ext)s"),
                "quiet": False,
                "overwrites": True,
                # Skip postprocessors until ffmpeg path is configured
                # "postprocessors": [
                #     {
                #         "key": "FFmpegExtractAudio",
                #         "preferredcodec": "mp3",
                #         "preferredquality": "192",
                #     },
                #     {
                #         "key": "FFmpegMetadata",
                #     },
                # ],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print("Successfully downloaded '" + title + "'")
        except yt_dlp.utils.DownloadError as e:
            print("Download error for '" + title + "': " + str(e))
        except Exception as e:  # pylint: disable=broad-except
            print("Unexpected error for '" + title + "': " + str(e))


def main():
    """Main function to run the download process."""
    download_songs()
    print("\nAll song downloads complete!")


if __name__ == "__main__":
    main()
