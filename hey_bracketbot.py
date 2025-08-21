# /// script
# dependencies = [
#   "bbos",
#   "metaphone",
#   "textdistance",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/BracketBotOS", editable = true }
# ///
from bbos import Reader, Writer, Type, Config
from bbos.app_manager import start_app, stop_app
import metaphone
import textdistance
import numpy as np
import subprocess
import socket

CFG = Config("transcriber")
CFG_LED_STRIP = Config("led_strip")

def phonetic(token):
    return metaphone.doublemetaphone(token)[0]   # primary code

def detect_wake_word(text, target_word):
    """Detect wake words using phonetic matching"""
    if not text:
        return False
    
    # Split into words and filter out empty strings
    words = [w.strip().lower() for w in text.split() if w.strip()]
    
    # Get phonetic code for target word
    target_code = phonetic(target_word)
    
    # Check if any word in the text matches the target word phonetically
    for word in words:
        word_code = phonetic(word)
        similarity = 1 - textdistance.levenshtein.normalized_distance(word_code, target_code)
        if similarity > 0.7:  # Threshold for phonetic match
            print(f"Detected '{target_word}' as '{word}' (similarity: {similarity:.2f})", flush=True)
            return True
        print(f"No match found for '{target_word}' in '{word}' (similarity: {similarity:.2f})", flush=True)
    
    return False

def speak_hostname():
    """Get the hostname and speak it using kokoro TTS"""
    try:
        # Get the hostname
        hostname = socket.gethostname()
        print(f"Speaking hostname: {hostname}", flush=True)
        
        subprocess.Popen([
            "uv", "run",
            "/home/bracketbot/BracketBotApps/kokoro/main.py", 
            hostname
        ], cwd="/home/bracketbot/BracketBotApps/kokoro")
        return True
    except Exception as e:
        print(f"Error speaking hostname: {e}", flush=True)
        return False

def main():
    with Reader("transcript") as r_transcript, \
         Writer("led_strip.ctrl", Type('led_strip_ctrl')) as w_led_strip:
        rgb_array = np.zeros((CFG_LED_STRIP.num_leds,3), dtype=np.uint8)
        color = [255, 255, 255]
        while True:
            if r_transcript.ready():
                text = r_transcript.data['text']
                if text:
                    if detect_wake_word(text, "follow"):
                        start_app("follow")
                        color = [0, 255, 0]
                    if detect_wake_word(text, "stop"):
                        stop_app("follow")
                        color = [255, 0, 0]
                    if detect_wake_word(text, "talk"):
                        start_app("realtime")
                        color = [255, 0, 255]
                    if detect_wake_word(text, "quiet"):
                        stop_app("realtime")
                        color = [0, 0, 0]
                    if detect_wake_word(text, "wave"):
                        start_app("mimic")
                        color = [255, 255, 0]
                    if detect_wake_word(text, "number"):
                        speak_hostname()
                        color = [0, 255, 255]  # Cyan color for number speaking
            rgb_array[:, :] = color
            w_led_strip["rgb"] = rgb_array
    stop_app("follow")

if __name__ == "__main__":
    main()