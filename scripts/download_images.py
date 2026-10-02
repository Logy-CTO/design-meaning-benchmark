"""Download and verify the eight image archives from the fixed GitHub release."""
from pathlib import Path
import hashlib,urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[1]
BASE='https://github.com/Logy-CTO/design-meaning-benchmark/releases/download/v1.0.1/'
HASHES={'images_broom.zip': 'ad4c8dec5e28bc88eb770e3cc188a81844363bfcdb6cafe5fd3cfa718353f5bd', 'images_desk_lamp.zip': 'd414872cbfd03266fa9c23ea23506a3133b3227d3c31e12172228c41f9fbf3eb', 'images_early_automobile.zip': 'b478f270b147d21f529619d51a1aa6075053e6ee14c61541e042d08f1ed15d77', 'images_electric_kettle.zip': 'e7b4873aee2484a91002e1f6ad07ec166f2dfe8ca6a448d6ec76be9edda13dce', 'images_lounge_chair.zip': '4fcc960a54644601528fe7a2de31066a2fd7c10a2459b99c892f2d57199552ff', 'images_sneakers.zip': '06cc81e023288c7f37dd1f3c8a71d5db9af32e86c6eb7272d4f7762c513c39ac', 'images_sports_car.zip': 'b27c6e8a55235145c0b6d6ff220d9ce6085b29d1ad62e2d0c9c735d0c1b28a99', 'images_vacuum_cleaner.zip': 'dea6c5929b221e9d8c2fe16c0af4339a3141a3cfeda1cd0902881ed9aaeb6e2e'}
cache=ROOT/'downloaded_archives';cache.mkdir(exist_ok=True)
for filename,digest in sorted(HASHES.items()):
    path=cache/filename
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        print('Downloading',filename,flush=True)
        request=urllib.request.Request(BASE+filename,headers={'User-Agent':'DesignMeaningBenchmark/1.0.1'})
        partial=path.with_suffix('.part')
        with urllib.request.urlopen(request,timeout=120) as src,partial.open('wb') as dst:
            while chunk:=src.read(1024*1024):dst.write(chunk)
        if hashlib.sha256(partial.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('SHA-256 mismatch: '+filename)
        partial.replace(path)
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:raise RuntimeError('ZIP corruption: '+filename)
        for member in archive.infolist():
            target=(ROOT/member.filename).resolve()
            if not target.is_relative_to(ROOT/'images'):raise RuntimeError('Invalid archive path')
        archive.extractall(ROOT)
    print('Verified and extracted',filename,flush=True)
print('All 1,600 image files are available under images/.')
