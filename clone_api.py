import io, os, json, uuid, base64, tempfile, ssl
ssl._create_default_https_context = ssl._create_unverified_context
from pathlib import Path
from typing import Optional
import torch
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from pydantic import BaseModel
from omnivoice import OmniVoice

app = FastAPI(title="Sharx OmniVox TTS & Clone API", version="1.0.0")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
VOICES_DIR = Path("./voices")
VOICES_DB = VOICES_DIR / "voices.json"
VOICES_DIR.mkdir(parents=True, exist_ok=True)

model = OmniVoice.from_pretrained("./models", device_map=DEVICE, dtype=torch.float16)

def load_db() -> dict:
    return json.loads(VOICES_DB.read_text(encoding="utf-8")) if VOICES_DB.exists() else {}

def save_db(data: dict):
    VOICES_DB.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def synth_wav_bytes(text: str, ref_audio: Optional[str] = None, ref_text: Optional[str] = None) -> bytes:
    kwargs = {"text": text}
    if ref_audio:
        kwargs["ref_audio"] = ref_audio
    if ref_text:
        kwargs["ref_text"] = ref_text
    audio = model.generate(**kwargs)
    wav_arr = audio[0].cpu().numpy() if torch.is_tensor(audio[0]) else audio[0]
    buf = io.BytesIO()
    sf.write(buf, wav_arr, 24000, format="WAV")
    return buf.getvalue()

class TTSReq(BaseModel):
    text: str
    voice_id: Optional[str] = None
    ref_text: Optional[str] = None
    return_base64: bool = False

class CloneReq(BaseModel):
    text: str
    ref_audio_base64: str
    ref_text: Optional[str] = None

@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "device": DEVICE, "voices_count": len(load_db())}

@app.get("/api/v1/voices")
async def list_voices():
    db = load_db()
    return {"voices": [{"voice_id": k, **v} for k, v in db.items()]}

@app.post("/api/v1/voices/upload")
async def upload_voice(
    file: UploadFile = File(...),
    name: str = Form(...),
    voice_id: Optional[str] = Form(None),
    ref_text: Optional[str] = Form(None),
):
    vid = (voice_id or name).strip().lower().replace(" ", "_") or uuid.uuid4().hex[:8]
    wav_path = VOICES_DIR / f"{vid}.wav"
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio file")
    wav_path.write_bytes(raw)
    db = load_db()
    db[vid] = {"name": name, "file": str(wav_path), "ref_text": ref_text}
    save_db(db)
    return {"status": "uploaded", "voice_id": vid, "voice": db[vid]}

@app.delete("/api/v1/voices/{voice_id}")
async def delete_voice(voice_id: str):
    db = load_db()
    if voice_id not in db:
        raise HTTPException(status_code=404, detail="voice_id not found")
    p = Path(db[voice_id]["file"])
    if p.exists():
        p.unlink()
    del db[voice_id]
    save_db(db)
    return {"status": "deleted", "voice_id": voice_id}

@app.post("/api/v1/tts")
async def generate_tts(req: TTSReq):
    ref_audio = None
    ref_text = req.ref_text
    if req.voice_id:
        db = load_db()
        if req.voice_id not in db:
            raise HTTPException(status_code=404, detail=f"voice_id '{req.voice_id}' not found")
        ref_audio = db[req.voice_id]["file"]
        if not ref_text:
            ref_text = db[req.voice_id].get("ref_text")
    try:
        wav_bytes = synth_wav_bytes(req.text, ref_audio=ref_audio, ref_text=ref_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if req.return_base64:
        return {"status": "success", "audio_base64": base64.b64encode(wav_bytes).decode("utf-8")}
    return Response(content=wav_bytes, media_type="audio/wav")

@app.post("/api/v1/audio/clone")
@app.post("/v1/audio/clone")
async def clone_voice(req: CloneReq):
    tmp_wav = None
    try:
        fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
        with os.fdopen(fd, "wb") as fb:
            fb.write(base64.b64decode(req.ref_audio_base64))
        wav_bytes = synth_wav_bytes(req.text, ref_audio=tmp_wav, ref_text=req.ref_text)
        return {"status": "success", "audio_base64": base64.b64encode(wav_bytes).decode("utf-8")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.remove(tmp_wav)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3033)
