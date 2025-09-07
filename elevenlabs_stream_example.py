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

# Generate speech with streaming
print("Generating speech...")
audio_stream = client.text_to_speech.stream(
    text="y othis is not as fast as it seem homie",
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id="eleven_flash_v2_5",
    output_format="pcm_16000"  # Raw audio at 16kHz
)

# Process and play audio chunks as they arrive
with Writer("speakerphone.speaker", Type("speakerphone_speaker")) as speaker:
    audio_buffer = np.array([], dtype=np.int16)
    
    for chunk in audio_stream:
        if chunk:
            # Convert chunk to numpy array and add to buffer
            chunk_data = np.frombuffer(chunk, dtype=np.int16)
            audio_buffer = np.concatenate([audio_buffer, chunk_data])
            
            # Process complete chunks from buffer
            while len(audio_buffer) >= CFG.speaker_chunk_size:
                with speaker.buf() as buf:
                    buf['audio'] = audio_buffer[:CFG.speaker_chunk_size].reshape(-1, 1)
                audio_buffer = audio_buffer[CFG.speaker_chunk_size:]
    
    # Handle remaining data in buffer
    if len(audio_buffer) > 0:
        last_chunk = np.zeros((CFG.speaker_chunk_size, 1), dtype=np.int16)
        last_chunk[:len(audio_buffer), 0] = audio_buffer
        with speaker.buf() as buf:
            buf['audio'] = last_chunk

print("Done!")
