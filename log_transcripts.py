# /// script
# dependencies = [
#   "bbos",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/BracketBotOS", editable = true }
# ///
from bbos import Reader
import time
from datetime import datetime

if __name__ == "__main__":
    log_file = "transcripts.log"
    
    with Reader("transcript") as r_transcript:
        print(f"Starting transcript logger. Logging to {log_file}")
        
        while True:
            if r_transcript.ready():
                text = str(r_transcript.data['text']).strip()
                
                if text:  # Only log non-empty transcripts
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_entry = f"[{timestamp}] {text}\n"
                    
                    # Append to log file
                    with open(log_file, 'a') as f:
                        f.write(log_entry)
                    
                    # Also print to console
                    print(log_entry.strip())
