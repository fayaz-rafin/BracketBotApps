# /// script
# dependencies = [
#   "bbos",
#   "kokoro @ git+https://github.com/hexgrad/kokoro.git",
#   "scipy",
#   "numpy",
#   "torch",
#   "transformers",
#   "phonemizer",
#   "pip",
#   "spacy",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/BracketBotOS", editable = true }
# ///
"""
Kokoro Text-to-Speech app for BracketBotOS.
Generates speech from text and plays it through the speaker daemon.
"""
import time
import sys
import os
import hashlib

script_start = time.time()

def get_cache_path(text, voice='am_adam'):
    """Generate a cache file path based on text and voice."""
    cache_dir = os.path.join(os.path.dirname(__file__), '.tts_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a hash of the text and voice for the filename
    text_hash = hashlib.md5(f"{text}_{voice}".encode()).hexdigest()[:16]
    # Make filename somewhat readable
    safe_text = "".join(c if c.isalnum() else "_" for c in text[:20])
    filename = f"{safe_text}_{text_hash}.wav"
    
    return os.path.join(cache_dir, filename)

def play_cached_audio(cache_path):
    """Play cached audio without importing heavy dependencies."""
    print(f"[CACHE HIT] Playing cached audio from: {cache_path}")
    
    # Import only what we need for playback
    from scipy.io import wavfile
    import numpy as np
    from bbos import Writer, Config, Type
    
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

def generate_and_play_audio(text, voice='am_adam', cache_path=None):
    """Generate audio using kokoro and play it."""
    print("[CACHE MISS] Generating new audio...")
    
    # NOW we import the heavy stuff only when needed
    print("Loading Kokoro dependencies...")
    
    # Monkey patch before any imports
    import phonemizer.backend.espeak.wrapper
    phonemizer.backend.espeak.wrapper.EspeakWrapper.set_data_path = lambda path: None
    phonemizer.backend.espeak.wrapper.EspeakWrapper.set_library = lambda path: None
    
    from kokoro.pipeline import KPipeline
    import numpy as np
    from scipy import signal
    from scipy.io import wavfile
    from bbos import Writer, Config, Type
    
    CFG = Config("speakerphone")
    
    def resample_audio(audio, orig_sr=24000, target_sr=16000):
        """Resample audio from original sample rate to target sample rate."""
        if orig_sr == target_sr:
            return audio
        
        resample_ratio = target_sr / orig_sr
        num_samples = int(len(audio) * resample_ratio)
        resampled = signal.resample(audio, num_samples)
        return resampled
    
    print("Initializing Kokoro pipeline...")
    pipeline = KPipeline(lang_code='a')
    
    print(f"Generating speech for: '{text}'")
    start_time = time.time()
    
    # Collect audio chunks
    audio_chunks = []
    for gs, ps, audio in pipeline(text, voice=voice):
        audio_chunks.append(audio.detach().cpu().numpy())
    
    audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
    
    if hasattr(audio, 'detach'):
        audio = audio.detach().cpu().numpy()
    
    # Flatten if multi-dimensional
    if len(audio.shape) > 1:
        audio = audio.flatten()
    
    elapsed_time = time.time() - start_time
    print(f"Audio generation took {elapsed_time:.2f} seconds.")
    
    # Resample from 24kHz to 16kHz (speaker daemon expects 16kHz)
    print(f"Resampling from 24000Hz to {CFG.speaker_sample_rate}Hz...")
    audio_resampled = resample_audio(audio, 24000, CFG.speaker_sample_rate)
    
    # Convert float32 [-1, 1] to int16
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
    sample_text = "Hello World, I'm Bracket Bot, and I generated this audio entirely on my own."
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
    else:
        text = sample_text
    
    voice = 'am_adam'
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