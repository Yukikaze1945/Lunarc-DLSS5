"""Prepare small component installer assets; retain legacy splitting for reproduction.

No recompression or modification of the accepted installer/payload is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def describe(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return dict(name=path.name, bytes=path.stat().st_size, sha256=digest.hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--version', default='1.4.0')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    setup = args.input / f'Lunarc-{args.version}-x64-Setup.exe'
    payload = args.input / f'lunarc-workbench-{args.version}-x64.nsis.7z'
    shutil.copy2(setup, args.output / setup.name)
    if not payload.is_file():
        info=dict(schema=1,mode='components',version=args.version,installer=describe(setup))
        (args.output/'INSTALLER.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
        shutil.copy2(Path(__file__).resolve().parents[1]/'docs/installing.md',args.output/'INSTALL.md')
        entries=[describe(p) for p in sorted(args.output.iterdir()) if p.is_file()]
        (args.output/'SHA256SUMS.txt').write_text(''.join(f'{e["sha256"]}  {e["name"]}\n' for e in entries),encoding='utf-8')
        print(json.dumps(info,indent=2))
        return
    parts = []
    limit = 1536 * 1024 * 1024  # 1.5 GiB, strictly below GitHub's 2 GiB limit.
    with payload.open('rb') as reader:
        remaining = payload.stat().st_size
        while remaining:
            part = args.output / f'{payload.name}.{len(parts) + 1:03d}'
            part_remaining = min(limit, remaining)
            with part.open('wb') as writer:
                while part_remaining:
                    data = reader.read(min(8 * 1024 * 1024, part_remaining))
                    if not data:
                        raise RuntimeError('Input payload ended unexpectedly')
                    writer.write(data)
                    part_remaining -= len(data)
                    remaining -= len(data)
            parts.append(describe(part))
    manifest = dict(schema=1, version=args.version, installer=describe(setup),
                    payload=describe(payload), parts=parts)
    (args.output / 'installer-payload.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    for name in ('Install-Lunarc.cmd', 'Install-Lunarc.ps1'):
        shutil.copy2(Path(__file__).parent / name, args.output / name)
    rows = [describe(p) for p in sorted(args.output.iterdir()) if p.is_file()]
    (args.output / 'SHA256SUMS.txt').write_text(
        ''.join(f"{row['sha256']}  {row['name']}\n" for row in rows), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
