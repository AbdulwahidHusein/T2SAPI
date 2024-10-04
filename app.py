import subprocess
from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import re

app = FastAPI()

# Load the Amharic TTS model and tokenizer once
model = VitsModel.from_pretrained("facebook/mms-tts-amh")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-amh")

# Function to romanize Amharic text using uroman
def romanize_text(text):
    # Keep numbers intact and only romanize the Amharic characters
    amharic_text = re.sub(r'\d+', lambda x: x.group(0), text)  # Keep numbers
    # Call the romanization tool on the non-number characters
    process = subprocess.run(
        ["uroman", amharic_text],
        capture_output=True,
        text=True
    )
    return process.stdout.strip()

# Function to generate speech and return the audio file
def generate_amharic_tts(text):
    # Romanize the Amharic text
    romanized_text = romanize_text(text)

    # Tokenize the romanized text
    inputs = tokenizer(romanized_text, return_tensors="pt")

    # Generate the waveform
    with torch.no_grad():
        output = model(**inputs).waveform

    # Save the waveform as a .wav file
    output_file = "amharic_tts_output.wav"
    scipy.io.wavfile.write(output_file, rate=model.config.sampling_rate, data=output[0].cpu().numpy())
    return output_file

class TTSRequest(BaseModel):
    text: str

@app.post("/api/generate-tts")
async def generate_tts_api(request: TTSRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")

    output_file = generate_amharic_tts(request.text)

    return FileResponse(output_file, media_type='audio/wav', filename="amharic_tts_output.wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
