"""Pinned public checkpoint, concurrent HTTP ranges, SHA256 verified."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import subprocess

REPO='bayliner1980/Qwen2.5-0.5B-MoE-A0.3B'
REV='4b89028ef7376d0e92acbbc4a7f560c2374aeec0'
SHA='6c1673e92522af47907eb8e1825142061a367ec5993e4060554fbce271f683be'
SIZE=989730216
root=Path('models/qwen_moe');root.mkdir(parents=True,exist_ok=True)
parts=Path('tmp/qwen_parts');parts.mkdir(parents=True,exist_ok=True)
url=f'https://hf-mirror.com/{REPO}/resolve/{REV}/model.safetensors?download=true'
chunk=4*1024*1024

def fetch(index):
    begin=index*chunk;end=min(SIZE,begin+chunk)-1
    dest=parts/str(index)
    if dest.exists() and dest.stat().st_size==end-begin+1:return
    for attempt in range(6):
        existing=dest.stat().st_size if dest.exists() else 0
        if existing==end-begin+1:return
        fragment=parts/f'{index}.fragment'
        result=subprocess.run(['curl','-L','--fail','--silent','--show-error','--range',f'{begin+existing}-{end}','--max-time','120','-o',str(fragment),url])
        if fragment.exists():
            data=fragment.read_bytes()
            if len(data)>end-begin+1-existing:raise RuntimeError('server ignored range')
            with dest.open('ab') as f:f.write(data)
            fragment.unlink()
        if dest.stat().st_size==end-begin+1:return
    raise RuntimeError(f'download range {index} failed')

target=root/'model.safetensors'
verified=False
if target.exists() and target.stat().st_size==SIZE:
    digest=hashlib.sha256()
    with target.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):digest.update(data)
    verified=digest.hexdigest()==SHA
if not verified:
    with ThreadPoolExecutor(max_workers=64) as pool:
        futures=[pool.submit(fetch,i) for i in range((SIZE+chunk-1)//chunk)]
        for count,f in enumerate(as_completed(futures),1):
            f.result()
            if count%24==0:print(f'{count}/{len(futures)} ranges',flush=True)
    sha=hashlib.sha256()
    with (root/'model.tmp').open('wb') as out:
        for i in range((SIZE+chunk-1)//chunk):
            data=(parts/str(i)).read_bytes();sha.update(data);out.write(data)
    if sha.hexdigest()!=SHA:raise RuntimeError(f'checksum mismatch: {sha.hexdigest()}')
    (root/'model.tmp').replace(root/'model.safetensors')
(root/'SOURCE.txt').write_text(f'{REPO}\nrevision={REV}\nsha256={SHA}\n')
for filename in ('config.json',):
    if (root/filename).exists():continue
    subprocess.run(['curl','-L','--fail','--silent','--retry','2','--max-time','90','-o',str(root/filename),f'https://huggingface.co/{REPO}/resolve/{REV}/{filename}'],check=True)
print('Verified model download complete',flush=True)
