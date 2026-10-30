import os
# Cloud edition: model downloads are permitted; images are processed on this server.
os.environ.pop('HF_HUB_OFFLINE',None)
os.environ.pop('TRANSFORMERS_OFFLINE',None)
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
from pathlib import Path
from functools import lru_cache
import hashlib
import io
import json
import uuid
import threading
import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageFilter

Image.MAX_IMAGE_PIXELS=25_000_000

def read_image(source):
    if isinstance(source,bytes):
        if len(source)>20*1024*1024: raise ValueError('Image exceeds 20 MB.')
        source=io.BytesIO(source)
    with Image.open(source) as im:
        if im.width*im.height>25_000_000: raise ValueError('Image exceeds 25 megapixels.')
        result=ImageOps.exif_transpose(im).convert('RGB')
        result.thumbnail((1600,1600))
        return result.copy()

def png_bytes(im):
    out=io.BytesIO(); im.save(out,format='PNG'); return out.getvalue()

def normalize(v):
    v=np.asarray(v,dtype=np.float32).ravel()
    return v/max(float(np.linalg.norm(v)),1e-8)

def basic_features(im):
    """Handcrafted demonstration descriptor; no labels or metadata used."""
    a=np.asarray(im.resize((128,128)),dtype=float)/255
    g=a.mean(axis=2)
    background=np.asarray(Image.fromarray(np.uint8(g*255)).filter(ImageFilter.GaussianBlur(12)),dtype=float)/255
    residual=g-background
    dy,dx=np.gradient(g)
    magnitude=np.hypot(dx,dy)
    angle=np.arctan2(dy,dx)
    orientation=np.histogram(angle,bins=16,range=(-np.pi,np.pi),weights=magnitude)[0]
    contrast=np.histogram(residual,bins=24,range=(-.6,.6))[0]
    mask=np.uint8(np.clip((np.median(g)-g)*4,0,1)*255)
    spatial=np.asarray(Image.fromarray(mask).resize((16,16)),dtype=float).ravel()/255
    colors=np.concatenate([np.histogram(a[:,:,i],bins=16,range=(0,1))[0] for i in range(3)])
    # Radial power separates fine pits/edges from broad stains and dents.
    spectrum=np.abs(np.fft.fftshift(np.fft.fft2(g-g.mean())))**2
    yy,xx=np.mgrid[:128,:128]
    radius=np.sqrt((xx-64)**2+(yy-64)**2)
    power=np.array([spectrum[(radius>=lo)&(radius<hi)].sum() for lo,hi in zip([1,3,6,12,24,40],[3,6,12,24,40,91])])
    power=np.sqrt(power/max(power.sum(),1e-8))
    chroma=np.array([np.mean(a[:,:,0]-a[:,:,2])*20, np.std(g)*5, np.mean(magnitude)*15])
    return normalize(np.concatenate([normalize(spatial),normalize(orientation)*.5,normalize(contrast)*.3,normalize(colors)*.3,power*2,chroma*2]))

MODEL_ID='openai/clip-vit-base-patch32'
MODEL_REVISION='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'

@st.cache_resource(show_spinner=False)
def clip_model(folder=None):
    import torch
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    torch.set_num_threads(2)
    processor=CLIPImageProcessor.from_pretrained(MODEL_ID,revision=MODEL_REVISION)
    # The pinned official release supplies pytorch_model.bin. Restrict loading to weights.
    model=CLIPVisionModelWithProjection.from_pretrained(
        MODEL_ID,revision=MODEL_REVISION,use_safetensors=False,
        weights_only=True,trust_remote_code=False).eval()
    return processor,model,threading.RLock()

def embed(im,method,model_dir=None):
    if method=='Demo visual features': return basic_features(im)
    import torch
    processor,model,lock=clip_model()
    with lock,torch.inference_mode():
        out=model(**processor(images=im,return_tensors='pt')).image_embeds
    return normalize(out[0].cpu().numpy())

def pixel_hash(im):
    im=im.convert('RGB')
    return hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()

@lru_cache(maxsize=128)
def reference_feature(path,mtime,size,method):
    # Only bundled public images enter this shared cache; no uploaded image bytes.
    im=read_image(path)
    return pixel_hash(im),embed(im,method)

def load_cases(root):
    root=Path(root); file=root/'cases.json'
    if not file.exists(): return []
    records=json.loads(file.read_text(encoding='utf-8'))
    if not isinstance(records,list): raise ValueError('cases.json must contain a list.')
    return records

def image_path(root,row):
    if '_image_bytes' in row: return io.BytesIO(row['_image_bytes'])
    root=Path(root).resolve(); path=(root/row['image']).resolve()
    if not path.is_relative_to(root): raise ValueError('Image must be inside the case folder.')
    return path

def search(query,rows,root,method,model_dir=None,exclude_bytes=None):
    q=embed(query,method); found=[]; errors=[]
    exclude_hash=pixel_hash(read_image(exclude_bytes)) if exclude_bytes else None
    for row in rows:
        try:
            if '_image_bytes' in row:
                im=read_image(row['_image_bytes']); digest=pixel_hash(im)
                if digest==exclude_hash: continue
                vec=embed(im,method)
            else:
                path=image_path(root,row); stat=path.stat()
                digest,vec=reference_feature(str(path),stat.st_mtime_ns,stat.st_size,method)
                if digest==exclude_hash: continue
            found.append((float(np.clip(q@vec,-1,1)),row))
        except (OSError,ValueError,KeyError) as exc:
            errors.append(f"{row.get('case_id','Unknown')}: {exc}")
    return sorted(found,key=lambda x:x[0],reverse=True),errors

def add_session_case(rows,im,metadata):
    if len(rows)>=20: raise ValueError('Maximum 20 temporary references per session.')
    data=png_bytes(im)
    if sum(len(r['_image_bytes']) for r in rows)+len(data)>50*1024*1024:
        raise ValueError('Session reference limit is 50 MB. Clear session cases to start again.')
    cid='SESSION-'+uuid.uuid4().hex[:10]
    rows.append(dict(metadata,case_id=cid,_image_bytes=data,synthetic=False))
    return cid
