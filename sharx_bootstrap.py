import os
import subprocess
import sys

# Lock execution to the exact Python binary to avoid WebUI venv traps
PY_BIN = sys.executable

print("\n--- SHARX OMNIVOICE DEPLOYMENT ---")
print(f"Runtime: {PY_BIN}")

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# 1. Force upgrade core build tools
run_cmd(f'"{PY_BIN}" -m pip install --upgrade pip wheel setuptools')

# 2. Install audio dependencies (bypass source compilation)
run_cmd(f'"{PY_BIN}" -m pip install soxr --only-binary :all:')

# 3. Install API components and the hf-xet storage translator
run_cmd(f'"{PY_BIN}" -m pip install huggingface_hub fastapi uvicorn pydantic soundfile requests hf-xet')

# 4. Install the OmniVoice package from your local cloned directory
run_cmd(f'"{PY_BIN}" -m pip install -e .')

# 5. Build the aggressive SSL-bypassing model downloader
dl_script = """
import os
import requests
import warnings
from huggingface_hub import snapshot_download

warnings.filterwarnings('ignore')

# Monkey-patch to strip SSL verification globally
old_request = requests.Session.request
def new_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return old_request(self, method, url, **kwargs)
requests.Session.request = new_request

# Force traffic through the domestic mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("Pulling 2.45GB Voice Clone Model (Bypassing Firewalls)...")
snapshot_download(repo_id="k2-fsa/OmniVoice", local_dir="./models")
"""

with open("sharx_download.py", "w") as f:
    f.write(dl_script.strip())

# Execute the downloader
run_cmd(f'"{PY_BIN}" sharx_download.py')

# 6. Generate the custom Voice Clone API (Port 3033)
api_code = """
import io, base64, tempfile, os, torch, ssl
ssl._create_default_https_context = ssl._create_unverified_context
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from omnivoice import OmniVoice

app = FastAPI(title='Sharx Voice Clone API')
print('Loading OmniVoice into VRAM...')

# Detect CUDA environment
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = OmniVoice.from_pretrained('./models', device_map=DEVICE, dtype=torch.float16)

class CloneReq(BaseModel):
    text: str
    ref_audio_base64: str  # The voice sample to clone
    ref_text: Optional[str] = None # Optional transcript of the reference audio

@app.post('/v1/audio/clone')
async def clone_voice(req: CloneReq):
    try:
        kwargs = {'text': req.text}
        
        # Decode the reference audio payload
        fd, tmp_wav = tempfile.mkstemp(suffix='.wav')
        with os.fdopen(fd, 'wb') as fb:
            fb.write(base64.b64decode(req.ref_audio_base64))
        
        kwargs['ref_audio'] = tmp_wav
        if req.ref_text: 
            kwargs['ref_text'] = req.ref_text
            
        # Generate the cloned speech
        audio = model.generate(**kwargs)
        os.remove(tmp_wav)
        
        # Encode output
        buf = io.BytesIO()
        sf.write(buf, audio[0].cpu().numpy() if torch.is_tensor(audio[0]) else audio[0], 24000, format='WAV')
        buf.seek(0)
        
        return {
            'status': 'success', 
            'audio_base64': base64.b64encode(buf.read()).decode('utf-8')
        }
    except Exception as e:
        if 'tmp_wav' in locals() and os.path.exists(tmp_wav): 
            os.remove(tmp_wav)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=3033)
"""

with open("clone_api.py", "w") as f:
    f.write(api_code.strip())

# 7. Launch the daemon
print("Booting Voice Clone Server on Port 3033...")
subprocess.Popen([PY_BIN, "clone_api.py"], stdout=open("api_boot.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
print("Run 'sys: tail -f api_boot.log' to monitor startup.")
