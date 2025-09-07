# /// script
# dependencies = [
#   "bbos",
#   "elevenlabs",
#   "python-dotenv"
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/BracketBotOS", editable = true }
# ///
import numpy as np
import os
from bbos import Writer, Config, Type
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

# Setup
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
CFG = Config("speakerphone")

# Generate speech
print("Generating speech...")
audio_stream = client.text_to_speech.convert(
    text="y othis is not as fast as it seem homie",
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id="eleven_flash_v2_5",
    output_format="pcm_16000"  # Raw audio at 16kHz
)

# Convert stream to numpy array
audio_data = np.frombuffer(b''.join(audio_stream), dtype=np.int16)

# Play audio in chunks
with Writer("speakerphone.speaker", Type("speakerphone_speaker")) as speaker:
    # Process full chunks
    for i in range(0, len(audio_data) - CFG.speaker_chunk_size, CFG.speaker_chunk_size):
        with speaker.buf() as buf:
            buf['audio'] = audio_data[i:i + CFG.speaker_chunk_size].reshape(-1, 1)
    
    # Handle last chunk with padding
    remaining = len(audio_data) % CFG.speaker_chunk_size
    if remaining > 0:
        last_chunk = np.zeros((CFG.speaker_chunk_size, 1), dtype=np.int16)
        last_chunk[:remaining, 0] = audio_data[-remaining:]
        with speaker.buf() as buf:
            buf['audio'] = last_chunk

print("Done!")
