"""Download and verify the exact published source snapshot for actual M4 fits."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib,json,time,urllib.request

root=Path(__file__).resolve().parents[1]/'data'
out=root/'archives'
out.mkdir(parents=True,exist_ok=True)
size=928675992
expected='243f48c505dc7320cfef8d22911d7fa82fa6f9a19e99132d29aa1d988ebff954'
url='https://ndownloader.figshare.com/files/57268553'
file=out/'Public_ScientificData_AHUFaults.zip'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
if file.exists() and file.stat().st_size==size and sha(file)==expected:
 print('Exact source archive already verified.',flush=True);raise SystemExit(0)
start=time.perf_counter()
chunk=32*1024*1024
parts=[(n,a,min(a+chunk,size)-1) for n,a in enumerate(range(0,size,chunk))]
def fetch(part):
 n,a,b=part;p=out/f'source_chunk_{n:03d}.partial'
 if p.exists() and p.stat().st_size==b-a+1:return p
 for attempt in range(3):
  try:
   req=urllib.request.Request(url,headers={'Range':f'bytes={a}-{b}','User-Agent':'AERIS-M4/1.0'})
   with urllib.request.urlopen(req,timeout=45) as r,p.open('wb') as f:
    if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {a}-{b}/{size}':raise ValueError(f'Unexpected byte range {r.headers.get("Content-Range")}')
    received=0
    while block:=r.read(1024*1024):
     received+=len(block)
     if received>b-a+1:raise ValueError('Oversized range')
     f.write(block)
   if p.stat().st_size!=b-a+1:raise ValueError('Truncated range')
   return p
  except Exception:
   if attempt==2:raise
with ThreadPoolExecutor(max_workers=6) as pool:
 futures={pool.submit(fetch,p):p[0] for p in parts}
 for f in as_completed(futures):
  p=f.result();print(f'Completed range {futures[f]+1}/{len(parts)} ({p.stat().st_size} bytes)',flush=True)
temp=file.with_suffix('.zip.partial')
with temp.open('wb') as f:
 for n,_,_ in parts:
  with (out/f'source_chunk_{n:03d}.partial').open('rb') as r:
   while block:=r.read(1024*1024):f.write(block)
if temp.stat().st_size!=size or sha(temp)!=expected:raise ValueError('Source snapshot checksum mismatch')
temp.replace(file)
for n,_,_ in parts:(out/f'source_chunk_{n:03d}.partial').unlink()
record={'url':url,'bytes':size,'sha256':expected,'action':'downloaded_uncached_and_verified','seconds':time.perf_counter()-start}
(out/'source_acquisition.json').write_text(json.dumps(record,indent=2))
print('Actual full source snapshot verified:',record,flush=True)
