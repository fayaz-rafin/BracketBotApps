# /// script
# dependencies = [
#   "bbos",
#   "yt-dlp",
#   "pydub",
#   "numpy",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/BracketBotOS", editable = true }
# ///
import sys
import tempfile
import os
import numpy as np
from pydub import AudioSegment
from bbos import Writer, Config, Type
import yt_dlp

CFG = Config("speakerphone")

def search_and_download_audio(search_query, output_path):
    """Search YouTube and download audio from the first result"""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
        'default_search': 'ytsearch',
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"[+] Searching YouTube for: {search_query}")
        try:
            # Search for the query and get the first result
            info = ydl.extract_info(f"ytsearch:{search_query}", download=False)
            if info['entries']:
                first_result = info['entries'][0]
                video_url = f"https://youtube.com/watch?v={first_result['id']}"
                title = first_result.get('title', 'Unknown')
                duration = first_result.get('duration', 0)
                
                print(f"[+] Found: {title}")
                print(f"[+] Duration: {duration // 60}:{duration % 60:02d}")
                print(f"[+] Downloading audio...")
                
                # Download the audio
                ydl.download([video_url])
                return True, title
            else:
                print("[-] No results found")
                return False, None
        except Exception as e:
            print(f"[-] Error downloading: {e}")
            return False, None

def convert_audio_for_speaker(input_path, output_path):
    """Convert audio to 16kHz mono WAV format"""
    print("[+] Converting audio format...")
    audio = AudioSegment.from_file(input_path)
    
    # Convert to mono if stereo
    if audio.channels > 1:
        audio = audio.set_channels(1)
    
    # Resample to 16kHz
    audio = audio.set_frame_rate(16000)
    
    # Export as WAV
    audio.export(output_path, format="wav")
    return len(audio) / 1000.0  # Duration in seconds

def play_audio(wav_path):
    """Play audio through the speaker"""
    print(f"[+] Playing audio...")
    
    # Load the WAV file
    audio = AudioSegment.from_wav(wav_path)
    samples = np.array(audio.get_array_of_samples())
    
    # Play through speaker
    with Writer("speakerphone.speaker", Type("speakerphone_speaker")) as w_speaker:
        chunk_count = 0
        chunk_size = CFG.speaker_chunk_size
        
        for i in range(0, len(samples), chunk_size):
            chunk = samples[i:i + chunk_size]
            
            # Pad the last chunk if necessary
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
            
            # Send to speaker
            with w_speaker.buf() as b:
                b['audio'] = chunk.reshape(-1, CFG.speaker_channels)
            
            chunk_count += 1
            
            # Print progress every second
            if chunk_count % 10 == 0:
                progress = (i / len(samples)) * 100
                print(f"\r[+] Progress: {progress:.1f}%", end='', flush=True)
        
        print(f"\n[+] Played {chunk_count} chunks")

def main():
    if len(sys.argv) < 2:
        print("Usage: python youtube_player.py \"search query\"")
        print("Example: python youtube_player.py \"lofi hip hop radio\"")
        sys.exit(1)
    
    search_query = " ".join(sys.argv[1:])
    
    # Create temporary directory for downloads
    with tempfile.TemporaryDirectory() as tmpdir:
        download_path = os.path.join(tmpdir, "audio")
        wav_path = os.path.join(tmpdir, "audio_converted.wav")
        
        # Search and download
        success, title = search_and_download_audio(search_query, download_path)
        if not success:
            sys.exit(1)
        
        # Find the downloaded file (yt-dlp adds .wav extension)
        downloaded_file = download_path + ".wav"
        if not os.path.exists(downloaded_file):
            # Sometimes it might be other formats
            for ext in ['.webm', '.m4a', '.mp3']:
                if os.path.exists(download_path + ext):
                    downloaded_file = download_path + ext
                    break
        
        if not os.path.exists(downloaded_file):
            print("[-] Error: Could not find downloaded audio file")
            sys.exit(1)
        
        # Convert to proper format
        duration = convert_audio_for_speaker(downloaded_file, wav_path)
        print(f"[+] Audio duration: {duration:.1f} seconds")
        
        # Play the audio
        try:
            play_audio(wav_path)
            print(f"\n[+] Finished playing: {title}")
        except KeyboardInterrupt:
            print("\n[!] Playback interrupted")
        except Exception as e:
            print(f"\n[-] Error during playback: {e}")

if __name__ == "__main__":
    main()
