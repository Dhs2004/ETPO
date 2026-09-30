"""Repair upstream WHEEL metadata, preserving binary contents and regenerating RECORD.

Decord's published py3-none wheel wrongly declares cp36-cp36m internally, although
its Python API uses ctypes. TextWorld puts a blank line before its Tag header,
which causes email-header parsers (including pip check) to ignore the tag.
"""
from pathlib import Path
import zipfile
from wheel.wheelfile import WheelFile

root = Path(__file__).resolve().parent / 'wheels'
output = root / 'metadata-fixed'
output.mkdir(exist_ok=True)
for source in (root / 'upstream').glob('*.whl'):
    tag = '-'.join(source.stem.split('-')[-3:])
    with zipfile.ZipFile(source) as archive, WheelFile(str(output / source.name), 'w') as target:
        for item in archive.infolist():
            if item.filename.endswith('.dist-info/RECORD'):
                continue
            data = archive.read(item.filename)
            if item.filename.endswith('.dist-info/WHEEL'):
                lines = [line for line in data.decode().splitlines() if line.strip() and not line.startswith('Tag:')]
                data = ('\n'.join(lines) + '\nTag: ' + tag + '\n').encode()
            target.writestr(item, data)
    print(output / source.name)
