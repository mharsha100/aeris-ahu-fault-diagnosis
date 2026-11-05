"""Extract exactly the47 hash-pinned original source/target input files."""
from pathlib import Path
import argparse,hashlib,zipfile,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SOURCE_SHA='243f48c505dc7320cfef8d22911d7fa82fa6f9a19e99132d29aa1d988ebff954'
TARGET_SHA='68c58568e3f0e4e7d0f0de3194a2ce7ca80703c057325ee402a94d3eab5a76c1'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def extract(source,target):
 if sha(source)!=SOURCE_SHA or sha(target)!=TARGET_SHA:raise ValueError('Pinned publisher archive hash mismatch')
 manifest=pd.read_csv(ROOT/'experiments/model_exploration/data_audit/input_manifest.csv');count=0
 with zipfile.ZipFile(source) as src,zipfile.ZipFile(target) as tgt:
  for row in manifest.itertuples():
   dest=(ROOT/'data'/row.relative_path).resolve()
   if not dest.is_relative_to((ROOT/'data').resolve()):raise ValueError('Invalid manifest path')
   if dest.exists():
    if sha(dest)!=row.sha256:raise ValueError('Existing input changed; preserve and investigate: '+str(dest))
    count+=1;continue
   archive=src if row.domain=='source' else tgt
   if row.domain=='source':
    suffix=Path(*Path(row.relative_path).parts[1:]).as_posix();matches=[n for n in archive.namelist() if n==suffix]
   else:matches=[n for n in archive.namelist() if Path(n).name==Path(row.relative_path).name]
   if len(matches)!=1:raise ValueError('Missing or ambiguous archived input: '+row.relative_path)
   content=archive.read(matches[0])
   if hashlib.sha256(content).hexdigest()!=row.sha256:raise ValueError('Raw member hash mismatch')
   dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content);count+=1
 print(f'Exactly{count} selected input files verified; unselected files are not treated as extra independent cases.')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-archive',type=Path,default=ROOT/'data/archives/Public_ScientificData_AHUFaults.zip');p.add_argument('--target-archive',type=Path,required=True);args=p.parse_args();extract(args.source_archive,args.target_archive)
