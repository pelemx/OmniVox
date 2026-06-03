import os
import subprocess

print("\n--- SHARX OMNIVOICE DEPLOYMENT ---")

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# 1. Bypass outdated GCC compiler and fix package states
run_cmd("python3 -m pip install --upgrade pip wheel setuptools")
run_cmd("python3 -m pip install soxr --only-binary :all:")

# 2. Install dependencies AND the crucial hf-xet package for Xet storage translation
run_cmd("python3 -m pip install huggingface_hub fastapi uvicorn pydantic soundfile requests hf-xet")
run_cmd("python3 -m pip install -e .")

# 3. Create a dedicated download script with an aggressive SSL bypass
dl_script = """
import os
import requests
import warnings
from huggingface_hub import snapshot_download

# Suppress the insecure request warnings
warnings.filterwarnings('ignore')

# Monkey-patch requests to globally disable SSL verification
old_request = requests.Session.request
def new_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return old_request(self, method, url, **kwargs)
requests.Session.request = new_request

# Force mirror routing
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("Downloading 2.45GB Xet-backed Model (Bypassing SSL Firewall)...")
snapshot_download(repo_id="k2-fsa/OmniVoice", local_dir="./models")
"""

with open("sharx_download.py", "w") as f:
    f.write(dl_script.strip())

# Execute the local downloader
run_cmd("python3 sharx_download.py")

# 4. Generate the custom port 3033 FastAPI backend
api_code = """
import io, base64, tempfile, os, torch, ssl
ssl._create_default_https_context = ssl._create_unverified_context
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from omnivoice import OmniVoice

app = FastAPI(title='Sharx OmniVoice TTS API')
print('Loading OmniVoice from local ./models directory...')
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = OmniVoice.from_pretrained('./models', device_map=DEVICE, dtype=torch.float16)

class TTSReq(BaseModel):
    text: str
    ref_audio_base64: Optional[str] = None
    ref_text: Optional[str] = None

@app.post('/v1/audio/speech')
async def gen_speech(req: TTSReq):
    try:
        kwargs = {'text': req.text}
        tmp = None
        if req.ref_audio_base64:
            fd, tmp = tempfile.mkstemp(suffix='.wav')
            with os.fdopen(fd, 'wb') as fb:
                fb.write(base64.b64decode(req.ref_audio_base64))
            kwargs['ref_audio'] = tmp
            if req.ref_text: kwargs['ref_text'] = req.ref_text
        audio = model.generate(**kwargs)
        if tmp and os.path.exists(tmp): os.remove(tmp)
        buf = io.BytesIO()
        sf.write(buf, audio[0].cpu().numpy() if torch.is_tensor(audio[0]) else audio[0], 24000, format='WAV')
        buf.seek(0)
        return {'status': 'success', 'audio_base64': base64.b64encode(buf.read()).decode('utf-8')}
    except Exception as e:
        if tmp and os.path.exists(tmp): os.remove(tmp)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=3033)
"""

with open("omni_api.py", "w") as f:
    f.write(api_code.strip())

# 5. Launch the daemon
print("Booting Server on Port 3033...")
subprocess.Popen(["python3", "omni_api.py"], stdout=open("omni_api.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
print("Deployment Complete. Run 'tail -f omni_api.log' to monitor boot status.")
