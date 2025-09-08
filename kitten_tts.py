# /// script
# dependencies = [
#   "bbos",
#   "scipy",
#   "numpy",
#   "soundfile",
#   "kittentts @ https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/BracketBotOS", editable = true }
# ///
"""
KittenTTS Text-to-Speech app for BracketBotOS.
Generates speech from text and plays it through the speaker daemon.
Based on KittenTTS - a lightweight TTS model under 25MB.
"""
import time
import sys
import os
import hashlib
import numpy as np
from scipy.io import wavfile
from scipy import signal
from bbos import Writer, Config, Type

script_start = time.time()

def get_cache_path(text, voice='expr-voice-2-f'):
    """Generate a cache file path based on text and voice."""
    cache_dir = os.path.join(os.path.dirname(__file__), '.kitten_tts_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a hash of the text and voice for the filename
    text_hash = hashlib.md5(f"{text}_{voice}".encode()).hexdigest()[:16]
    # Make filename somewhat readable
    safe_text = "".join(c if c.isalnum() else "_" for c in text[:20])
    filename = f"{safe_text}_{voice}_{text_hash}.wav"
    
    return os.path.join(cache_dir, filename)

def resample_audio(audio, orig_sr=24000, target_sr=16000):
    """Resample audio from original sample rate to target sample rate."""
    if orig_sr == target_sr:
        return audio
    
    resample_ratio = target_sr / orig_sr
    num_samples = int(len(audio) * resample_ratio)
    resampled = signal.resample(audio, num_samples)
    return resampled

def play_cached_audio(cache_path):
    """Play cached audio without importing heavy dependencies."""
    print(f"[CACHE HIT] Playing cached audio from: {cache_path}")
    
    CFG = Config("speakerphone")
    
    # Load cached audio
    sample_rate, audio_int16 = wavfile.read(cache_path)
    print(f"Loaded cached audio: {len(audio_int16)} samples at {sample_rate}Hz")
    
    # Send to speaker in chunks
    chunk_size = CFG.speaker_chunk_size
    total_samples = len(audio_int16)
    chunks_needed = (total_samples + chunk_size - 1) // chunk_size
    
    print(f"Playing audio ({total_samples} samples, {chunks_needed} chunks)...")
    
    with Writer("speakerphone.speaker", Type("speakerphone_speaker")) as w_speaker:
        for i in range(chunks_needed):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total_samples)
            chunk = audio_int16[start_idx:end_idx]
            
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
            
            chunk = chunk.reshape(-1, CFG.speaker_channels)
            
            with w_speaker.buf() as b:
                b['audio'] = chunk
    
    print(f"[CACHE] Playback complete in {time.time() - script_start:.2f}s total")
    return True

def generate_and_play_audio(text, voice='expr-voice-2-f', cache_path=None):
    """Generate audio using KittenTTS and play it."""
    print("[CACHE MISS] Generating new audio...")
    
    # Import KittenTTS when needed
    print("Loading KittenTTS model...")
    from kittentts import KittenTTS
    import soundfile as sf
    
    CFG = Config("speakerphone")
    
    # Initialize KittenTTS
    print("Initializing KittenTTS pipeline...")
    model = KittenTTS("KittenML/kitten-tts-nano-0.2")
    
    print(f"Generating speech for: '{text}' with voice: '{voice}'")
    start_time = time.time()
    
    # Generate audio
    audio = model.generate(text, voice=voice)
    
    elapsed_time = time.time() - start_time
    print(f"Audio generation took {elapsed_time:.2f} seconds.")
    
    # KittenTTS returns audio at 24kHz, resample to 16kHz for speaker daemon
    print(f"Resampling from 24000Hz to {CFG.speaker_sample_rate}Hz...")
    audio_resampled = resample_audio(audio, 24000, CFG.speaker_sample_rate)
    
    # Convert float32 to int16
    audio_int16 = np.clip(audio_resampled * 32768, -32768, 32767).astype(np.int16)
    
    # Save to cache if path provided
    if cache_path:
        print(f"Saving audio to cache: {cache_path}")
        wavfile.write(cache_path, CFG.speaker_sample_rate, audio_int16)
    
    # Send to speaker in chunks
    chunk_size = CFG.speaker_chunk_size
    total_samples = len(audio_int16)
    chunks_needed = (total_samples + chunk_size - 1) // chunk_size
    
    print(f"Playing audio ({total_samples} samples, {chunks_needed} chunks)...")
    
    with Writer("speakerphone.speaker", Type("speakerphone_speaker")) as w_speaker:
        for i in range(chunks_needed):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total_samples)
            chunk = audio_int16[start_idx:end_idx]
            
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
            
            chunk = chunk.reshape(-1, CFG.speaker_channels)
            
            with w_speaker.buf() as b:
                b['audio'] = chunk
    
    print(f"Total time: {time.time() - script_start:.2f}s")
    print("Done! Audio played successfully.")

def main():
    # Available voices from KittenTTS
    available_voices = [
        'expr-voice-2-m', 'expr-voice-2-f',
        'expr-voice-3-m', 'expr-voice-3-f',
        'expr-voice-4-m', 'expr-voice-4-f',
        'expr-voice-5-m', 'expr-voice-5-f'
    ]
    
    sample_text = "Hello World, I'm Bracket Bot, and I'm using Kitten TTS, a lightweight text to speech model."
    
    # Check for command line arguments
    voice = 'expr-voice-2-f'  # Default to female voice 2
    if len(sys.argv) > 1:
        # Check if first argument is a voice option
        if sys.argv[1] in available_voices:
            voice = sys.argv[1]
            text = ' '.join(sys.argv[2:]) if len(sys.argv) > 2 else sample_text
        else:
            text = ' '.join(sys.argv[1:])
    else:
        text = sample_text
        print(f"Usage: {sys.argv[0]} [voice] [text to speak]")
        print(f"Available voices: {', '.join(available_voices)}")
        print(f"Using default voice: {voice}")
    
    cache_path = get_cache_path(text, voice)
    
    # CHECK CACHE FIRST - before importing anything heavy!
    if os.path.exists(cache_path):
        # Fast path - just play the cached audio
        play_cached_audio(cache_path)
    else:
        # Slow path - generate new audio
        generate_and_play_audio(text, voice, cache_path)

if __name__ == "__main__":
    main()
