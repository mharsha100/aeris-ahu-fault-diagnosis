"""Verify cached publisher archives; download only when a cache file is absent."""
import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

SOURCES = [
    {"name":"Public_ScientificData_AHUFaults.zip", "doi":"10.6084/m9.figshare.29297999.v3",
     "url":"https://ndownloader.figshare.com/files/57268553", "license":"CC BY 4.0",
     "bytes":928675992, "sha256":"243f48c505dc7320cfef8d22911d7fa82fa6f9a19e99132d29aa1d988ebff954"},
    {"name":"lbnl_2019_afdd.zip", "doi":"10.25984/1824861", "release_updated":"2021-10-15",
     "url":"https://data.openei.org/files/910/Data%20Sets%20for%20AFDD%20Evauluation%20of%20Building%20FDD%20Algorithms.zip",
     "license":"CC BY 4.0", "bytes":7010416,
     "sha256":"68c58568e3f0e4e7d0f0de3194a2ce7ca80703c057325ee402a94d3eab5a76c1"},
]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--record',type=Path,default=Path('ingestion.json'))
    args=ap.parse_args(); records=[]
    for spec in SOURCES:
        path=args.cache/spec['name']; cached=path.exists()
        if not cached:
            args.cache.mkdir(parents=True,exist_ok=True)
            temp=path.with_suffix('.zip.partial')
            with urllib.request.urlopen(urllib.request.Request(spec['url'],headers={'User-Agent':'AERIS-M3/1.0'}),timeout=120) as response, temp.open('wb') as f:
                while chunk:=response.read(1024*1024): f.write(chunk)
            path_for_check=temp
        else:
            path_for_check=path
        h=hashlib.sha256()
        with path_for_check.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        if h.hexdigest()!=spec['sha256'] or path_for_check.stat().st_size!=spec['bytes']:
            raise ValueError('Publisher snapshot mismatch: '+spec['name'])
        with ZipFile(path_for_check) as archive:
            # Selected members receive CRC verification as they are read by pipeline.py.
            member_count=len(archive.namelist())
        if not cached: path_for_check.replace(path)
        records.append({**spec,'action':'verified_existing_cache' if cached else 'downloaded_and_verified','archive_members':member_count,'checked_utc':datetime.now(timezone.utc).isoformat()})
    args.record.parent.mkdir(parents=True,exist_ok=True)
    args.record.write_text(json.dumps(records,indent=2),encoding='utf8')
    print('Verified both pinned archives; record:',args.record)


if __name__=='__main__':main()
