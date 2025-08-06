#!/usr/bin/env python
import os
from pytube import YouTube

# --- Configuration ---
SONGS = {
    "U2 - One": "https://www.youtube.com/watch?v=ftjEcrrf7r0",
    "Bon Jovi - Always": "https://www.youtube.com/watch?v=9BMwcO6_hyA",
    "Foreigner - I Want to Know What Love Is": "https://www.youtube.com/watch?v=r3Pr1_v7hsw",
}
DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "website", "audio")


# --- Main Download Logic ---
def download_song(song_title, url):
    """Downloads a single song as an MP3."""
    try:
        print(f"Downloading '{song_title}'...")
        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True, file_extension="mp4").first()
        if not audio_stream:
            print(f"No audio stream found for '{song_title}'.")
            return

        # Download the file
        output_file = audio_stream.download(output_path=DOWNLOAD_DIR)

        # Rename to a clean, friendly name
        base, ext = os.path.splitext(output_file)
        new_file_name = "".join(
            c for c in song_title if c.isalnum() or c in " -_"
        ).rstrip()
        new_file_path = os.path.join(DOWNLOAD_DIR, f"{new_file_name}.mp3")

        # Check for existing file and remove if necessary
        if os.path.exists(new_file_path):
            print(f"'{new_file_name}.mp3' already exists. Replacing.")
            os.remove(new_file_path)

        os.rename(output_file, new_file_path)
        print(f"Successfully downloaded and saved to '{new_file_path}'")

    except Exception as e:
        print(f"Error downloading '{song_title}': {e}")


def main():
    """Downloads all the songs specified in the SONGS dictionary."""
    # Ensure the download directory exists
    if not os.path.exists(DOWNLOAD_DIR):
        print(f"Creating download directory: '{DOWNLOAD_DIR}'")
        os.makedirs(DOWNLOAD_DIR)

    # Download each song
    for title, url in SONGS.items():
        download_song(title, url)

    print("\nAll song downloads complete!")


if __name__ == "__main__":
    main()
