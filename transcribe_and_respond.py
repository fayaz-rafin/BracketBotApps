# /// script
# dependencies = [
#   "bbos",
#   "openai",
#   "python-dotenv"
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/BracketBotOS", editable = true }
# ///
from bbos import Reader
import time
from datetime import datetime
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Store transcript history and conversation with AI
    transcript_history = []
    conversation_history = []  # Includes both user messages and AI responses
    last_analysis_time = time.time()
    analysis_interval = 2.0  # seconds
    
    with Reader("transcript") as r_transcript:
        print("Starting transcript logger with OpenAI analysis...")
        print(f"Will analyze transcripts every {analysis_interval} seconds")
        print("-" * 50)
        
        while True:
            current_time = time.time()
            
            # Check for new transcripts
            if r_transcript.ready():
                text = str(r_transcript.data['text']).strip()
                
                if text:  # Only process non-empty transcripts
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    entry = f"[{timestamp}] {text}"
                    transcript_history.append(entry)
                    
                    # Add to conversation history with "User:" prefix
                    conversation_entry = f"[{timestamp}] User: {text}"
                    conversation_history.append(conversation_entry)
                    
                    print(f"New transcript: {entry}")
            
            # Check if it's time to analyze with OpenAI
            if current_time - last_analysis_time >= analysis_interval and transcript_history:
                last_analysis_time = current_time
                
                # Create conversation context including previous AI responses
                full_conversation = "\n".join(conversation_history)
                
                # Create prompt for OpenAI
                prompt = f"""{full_conversation}"""
                
                try:
                    # Get OpenAI response
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are given a transcript of a person talking to a robot live. The history includes both user messages (marked 'User:') and your previous responses (marked 'AI:'). Analyze if there's something new from the user that you haven't responded to yet. If yes, respond with 'YES: [your response]'. If no (you already responded or nothing new), just respond with 'NO'."},
                            {"role": "user", "content": prompt}
                        ],
                    )
                    
                    ai_response = response.choices[0].message.content.strip()
                    
                    # Add AI response to conversation history
                    ai_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ai_entry = f"[{ai_timestamp}] AI: {ai_response}"
                    conversation_history.append(ai_entry)
                    
                    print("\n" + "="*50)
                    print(f"OpenAI Analysis at {datetime.now().strftime('%H:%M:%S')}:")
                    print(ai_response)
                    print("="*50 + "\n")
                    
                except Exception as e:
                    print(f"\nError calling OpenAI API: {e}\n")
