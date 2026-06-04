import os, sys, io, base64, shutil, torch, ssl

os.environ['PATH'] += os.pathsep + '/u01/vt_media/miniconda3/envs/omni_env/bin'
ssl._create_default_https_context = ssl._create_unverified_context

import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
import uvicorn
from omnivoice import OmniVoice

app = FastAPI(title='Sharx OmniVoice API 0.4.0')
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = OmniVoice.from_pretrained('./models', device_map=DEVICE, dtype=torch.float16)

ROSTER_DIR = './roster'
os.makedirs(ROSTER_DIR, exist_ok=True)


def load_roster():
    roster = {}
    for filename in os.listdir(ROSTER_DIR):
        if filename.endswith('.wav'):
            voice_id = os.path.splitext(filename)[0]
            roster[voice_id] = os.path.join(ROSTER_DIR, filename)
    return roster


VOICE_ROSTER = load_roster()


class TTSRequest(BaseModel):
    text: str
    voice_id: str
    language: str = 'auto'
    reference_text: Optional[str] = None
    instruct: Optional[str] = None
    speed: float = 1.0
    duration: Optional[float] = None
    steps: int = 32
    cfg_scale: float = 2.0
    denoise: bool = True
    preprocess_prompt: bool = True
    postprocess_output: bool = True


@app.get('/api/v1/voices')
async def list_voices():
    return {'voices': list(VOICE_ROSTER.keys())}


@app.post('/api/v1/voices/upload')
async def upload_voice(voice_id: str = Form(...), file: UploadFile = File(...)):
    if not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail='Only .wav files are supported')
    file_path = os.path.join(ROSTER_DIR, f'{voice_id}.wav')
    with open(file_path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    VOICE_ROSTER[voice_id] = file_path
    return {'status': 'success', 'voice_id': voice_id, 'path': file_path}


@app.post('/api/v1/tts')
async def generate_tts(req: TTSRequest):
    if req.voice_id not in VOICE_ROSTER:
        raise HTTPException(status_code=400, detail='Invalid voice_id')

    kwargs = {
        'text': req.text,
        'ref_audio': VOICE_ROSTER[req.voice_id],
        'steps': req.steps,
        'cfg_scale': req.cfg_scale,
        'speed': req.speed,
        'denoise': req.denoise,
        'preprocess': req.preprocess_prompt,
        'postprocess': req.postprocess_output,
    }

    if req.language and req.language.lower() != 'auto':
        kwargs['language'] = req.language
    if req.reference_text and req.reference_text.strip():
        kwargs['ref_text'] = req.reference_text
    if req.instruct and req.instruct.strip():
        kwargs['instruct'] = req.instruct
    if req.duration is not None:
        kwargs['duration'] = req.duration

    try:
        audio = model.generate(**kwargs)
        buf = io.BytesIO()
        sf.write(
            buf,
            audio[0].cpu().numpy() if torch.is_tensor(audio[0]) else audio[0],
            24000,
            format='WAV'
        )
        buf.seek(0)
        return {
            'status': 'success',
            'audio_base64': base64.b64encode(buf.read()).decode('utf-8')
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=3249)
