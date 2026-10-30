import os
# Search never downloads model files. Only download_model.py may access the model hub.
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
from pathlib import Path
from functools import lru_cache
import hashlib
import io
import json
import uuid
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

@lru_cache(maxsize=2)
def clip_model(folder):
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    path=Path(folder)
    if not (path/'config.json').exists():
        raise ValueError('CLIP model not installed. Run download_model.py once while online, then restart the app.')
    processor=CLIPImageProcessor.from_pretrained(path,local_files_only=True)
    model=CLIPVisionModelWithProjection.from_pretrained(path,local_files_only=True,use_safetensors=True).eval()
    return processor,model

def embed(im,method,model_dir):
    if method=='Demo visual features': return basic_features(im)
    import torch
    processor,model=clip_model(str(model_dir))
    with torch.inference_mode():
        out=model(**processor(images=im,return_tensors='pt')).image_embeds
    return normalize(out[0].cpu().numpy())

@lru_cache(maxsize=256)
def cached_embed(data,method,model_dir):
    return embed(read_image(data),method,model_dir)

def load_cases(root):
    root=Path(root); file=root/'cases.json'
    if not file.exists(): return []
    records=json.loads(file.read_text(encoding='utf-8'))
    if not isinstance(records,list): raise ValueError('cases.json must contain a list.')
    return records

def image_path(root,row):
    root=Path(root).resolve(); path=(root/row['image']).resolve()
    if not path.is_relative_to(root): raise ValueError('Image must be inside the case folder.')
    return path

def search(query,rows,root,method,model_dir,exclude_bytes=None):
    q=embed(query,method,model_dir); found=[]; errors=[]
    exclude_hash=hashlib.sha256(exclude_bytes).hexdigest() if exclude_bytes else None
    for row in rows:
        try:
            data=png_bytes(read_image(image_path(root,row)))
            if exclude_hash and hashlib.sha256(data).hexdigest()==exclude_hash: continue
            vec=cached_embed(data,method,str(model_dir))
            found.append((float(np.clip(q@vec,-1,1)),row))
        except (OSError,ValueError,KeyError) as exc:
            errors.append(f"{row.get('case_id','Unknown')}: {exc}")
    return sorted(found,key=lambda x:x[0],reverse=True),errors

def add_case(root,im,metadata):
    root=Path(root); (root/'images').mkdir(parents=True,exist_ok=True)
    rows=load_cases(root); cid='CASE-'+uuid.uuid4().hex[:10]
    path=f'images/{cid}.png'; im.save(root/path)
    row=dict(metadata,case_id=cid,image=path,synthetic=False)
    rows.append(row)
    tmp=root/'cases.tmp'
    tmp.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    tmp.replace(root/'cases.json')
    return cid
