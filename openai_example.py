# /// script
# dependencies = [
#   "bbos",
#   "openai",
#   "python-dotenv"
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/BracketBotOS", editable = true }
# ///
import numpy as np
import os
from bbos import Writer, Config, Type
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.responses.create(
  model="gpt-4o-mini",
  input="Tell me a three sentence bedtime story about a unicorn."
)

print(response)
