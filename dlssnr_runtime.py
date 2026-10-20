"""User-owned DLSSNR runtime cache. Inspection never loads executable code.

No runtime or download mirror is distributed with Lunarc. An approved source
is a user's trust decision, not a statement about NVIDIA redistribution rights.
"""
import base64
import hashlib
import http.client
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import ssl
import struct
import subprocess
import threading
import time
import uuid
import zipfile
from pathlib import PurePosixPath
from urllib.parse import urlsplit, urljoin

NAME = 'nvngx_dlssnr.dll'
MAX_BYTES = 512 * 1024 * 1024
_SHA = re.compile(r'^[a-f0-9]{64}$')
# Data only, independently checked against our signed local file and GPU gate.
# Reference: Overglaze docs/MODEL.md @ 4d49c0a; Feeder installer @ 34edfe1.
KNOWN_VERSIONS = {
    'e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e': {
        'version': '310.8.0.0', 'bytes': 165840496,
        'label': '310.8.0.0',
        'url': 'https://github.com/RankFTW/rhi-repo/releases/download/dlssnr-310.8.0/nvngx_dlssnr_310.8.0.zip',
        'archive_sha256': '388c0a7912e15ec911b9c9e11a692142b11fe387ddf2b637d8c358138fffb3ac',
        'source': 'Feeder 所用社区来源 · RankFTW/rhi-repo',
    }
}

# Identity allowlist only, not a download catalog or a claim of GPU validation.
# OpenNR profiles/compat-fp16-310.9.16.json pins this reference output.
COMMUNITY_VERSIONS = {
    'cdfa6c1ec784867182164c7836f324f087b5c9f374d0c9fe46f54999feabf3ed': {
        'version': '310.9.16.0', 'bytes': 276162560,
        'publisher': 'ShortFuse / OpenNR',
        'reference': 'https://github.com/clshortfuse/openNR/blob/main/profiles/compat-fp16-310.9.16.json',
    },
}
MODEL_AB_HASHES = frozenset(COMMUNITY_VERSIONS) | {
    # NVIDIA base input pinned independently by the same OpenNR profile.
    '2bc19bc09a7b89dd44f73e2ac0108fcc586e2154bdd0830f242f4f040e28789f',
}


class RuntimeErrorDetail(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def fail(code, message):
    raise RuntimeErrorDetail(code, message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_pe(path):
    with Path(path).open('rb') as stream:
        header = stream.read(64)
        if len(header) != 64 or header[:2] != b'MZ':
            fail('invalid_pe', '文件不是有效的 Windows DLL')
        offset = struct.unpack_from('<I', header, 60)[0]
        if offset < 64 or offset > MAX_BYTES - 26:
            fail('invalid_pe', 'DLL 的 PE 头无效')
        stream.seek(offset)
        pe = stream.read(26)
        if (len(pe) != 26 or pe[:4] != b'PE\0\0' or
                struct.unpack_from('<H', pe, 4)[0] != 0x8664 or
                not struct.unpack_from('<H', pe, 22)[0] & 0x2000 or
                struct.unpack_from('<H', pe, 24)[0] != 0x20b):
            fail('wrong_architecture', '需要 Windows x64 的 PE32+ DLL')


def windows_identity(path):
    if os.name != 'nt':
        fail('signature_unavailable', '数字签名信息需要 Windows 才能读取')
    # Literal environment input, never interpolate a user path into shell code.
    script = r'''
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Import-Module (Join-Path $PSHOME 'Modules/Microsoft.PowerShell.Security/Microsoft.PowerShell.Security.psd1')
$p = $env:LUNARC_INSPECT_DLL
$sig = Get-AuthenticodeSignature -LiteralPath $p
$v = (Get-Item -LiteralPath $p).VersionInfo
[pscustomobject]@{status=$sig.Status.ToString(); signer=$sig.SignerCertificate.Subject;
 thumbprint=$sig.SignerCertificate.Thumbprint; product=$v.ProductName;
 description=$v.FileDescription; version=$v.FileVersion} | ConvertTo-Json -Compress
'''
    powershell = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
    try:
        result = subprocess.run([str(powershell), '-NoProfile', '-NonInteractive',
            '-EncodedCommand', base64.b64encode(script.encode('utf-16le')).decode('ascii')],
            env={**os.environ, 'LUNARC_INSPECT_DLL': str(Path(path).resolve()),
                 'PSModulePath': str(powershell.parent / 'Modules')},
            capture_output=True, timeout=30,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode:
            fail('signature_unavailable', 'Windows 签名校验失败，请检查系统证书与文件状态')
        return json.loads(result.stdout.decode('utf-8-sig', errors='replace'))
    except (OSError, subprocess.TimeoutExpired, ValueError):
        fail('signature_unavailable', '无法读取数字签名信息')


def inspect(path, expected_sha=None, identity_checker=windows_identity):
    path = Path(path)
    if not path.is_file():
        fail('file_missing', '未找到 DLSSNR DLL，请重新选择文件')
    if not 128 <= path.stat().st_size <= MAX_BYTES:
        fail('file_size', 'DLL 大小无效或超过 512 MiB')
    check_pe(path)
    digest = sha256(path)
    if expected_sha and digest != expected_sha.lower():
        fail('hash_mismatch', 'SHA-256 不匹配，文件已拒绝')
    try:
        identity = identity_checker(path)
    except RuntimeErrorDetail as error:
        if error.code != 'signature_unavailable':
            raise
        # User-selected builds remain usable when Windows cannot identify them.
        identity = {'status': 'Unavailable'}
    if not isinstance(identity, dict):
        identity = {'status': 'Unavailable'}
    community = COMMUNITY_VERSIONS.get(digest)
    product = identity.get('product') or ''
    numbers = re.findall(r'\d+', identity.get('version') or '')
    version_parts = tuple(int(v) for v in numbers) if len(numbers) == 4 and any(int(v) for v in numbers) else None
    version = '.'.join(map(str, version_parts)) if version_parts else 'unknown'
    pinned = bool(community and path.stat().st_size == community['bytes'] and
                  version in ('unknown', community['version']))
    nvidia = bool(identity.get('status') == 'Valid' and 'NVIDIA DLSSNR' in product and version_parts and
        re.search(r'(?:^|,\s*)(?:CN|O)=NVIDIA Corporation(?:,|$)', identity.get('signer') or '', re.I))
    trust = 'pinned_community' if pinned else 'nvidia_signature' if nvidia else 'unrecognized'
    if pinned and version == 'unknown':
        version = community['version']
    # Ensure inspection and the returned hash describe the same bytes.
    if sha256(path) != digest:
        fail('file_changed', '校验过程中 DLL 被修改，请重试')
    return {'id': digest, 'sha256': digest, 'bytes': path.stat().st_size,
            'compatibility': 'known' if nvidia and not pinned and digest in KNOWN_VERSIONS else 'uncharacterized',
            'version': version,
            'publisher': community['publisher'] if pinned else 'NVIDIA Corporation' if nvidia else identity.get('signer') or 'unknown',
            'signature': identity.get('status') or 'Unavailable',
            'trust': trust,
            'model_presets': [1, 2] if digest in MODEL_AB_HASHES or
                version_parts and version_parts >= (310, 9, 16, 0) else [1],
            'certificate': None if pinned else identity.get('thumbprint'),
            'product': product}


def parsed_url(url):
    try:
        value = urlsplit(url)
        if (value.scheme != 'https' or not value.hostname or value.username or
                value.password or value.fragment or value.port not in (None, 443)):
            raise ValueError()
        return value
    except ValueError:
        fail('invalid_url', '下载地址必须是无账号信息的 HTTPS 地址，使用标准 443 端口')


def public_address(host, allow_tun=False):
    try:
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError:
        fail('network_error', '无法解析下载服务器')
    ips = list(dict.fromkeys(item[4][0] for item in addresses))
    def allowed(ip):
        address = ipaddress.ip_address(ip)
        # Clash-style TUN DNS maps public hosts to the benchmarking IPv4 pool.
        # Only our exact reviewed catalog URL and GitHub asset hosts may use it;
        # HTTPS hostname verification and both pinned hashes still apply.
        return address.is_global or (allow_tun and address.version == 4 and
                                     address in ipaddress.ip_network('198.18.0.0/15'))
    if not ips or any(not allowed(ip) for ip in ips):
        fail('unsafe_address', '不允许从本机、内网或保留地址下载运行库')
    return ips[0]


def fetch_https(url, target, progress):
    original = parsed_url(url).hostname
    reviewed_url = any(entry.get('url') == url for entry in KNOWN_VERSIONS.values())
    deadline = time.monotonic() + 180
    for _ in range(5):
        value = parsed_url(url)
        github_redirect = original == 'github.com' and value.hostname in (
            'release-assets.githubusercontent.com', 'objects.githubusercontent.com')
        if value.hostname != original and not github_redirect:
            fail('unsafe_redirect', '下载重定向到了其他域名，请明确添加最终来源')
        tun_host = reviewed_url and value.hostname in ('github.com',
            'release-assets.githubusercontent.com', 'objects.githubusercontent.com')
        ip = public_address(value.hostname, allow_tun=True) if tun_host else public_address(value.hostname)
        connection = http.client.HTTPSConnection(value.hostname, timeout=20,
                                                  context=ssl.create_default_context())
        # Connect to the already checked IP, retaining original host for TLS and
        # certificate verification. Prevents DNS rebinding; ignores proxy env.
        connection._create_connection = lambda address, timeout, source_address=None: socket.create_connection((ip, 443), timeout, source_address)
        try:
            target_url = (value.path or '/') + ('?' + value.query if value.query else '')
            connection.request('GET', target_url, headers={'User-Agent': 'Lunarc-Runtime/1'})
            response = connection.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                url = urljoin(url, response.getheader('Location') or '')
                continue
            if response.status != 200:
                fail('http_error', f'运行库下载失败：HTTP {response.status}')
            length = response.getheader('Content-Length')
            total = int(length) if length else None
            if total is not None and not 128 <= total <= MAX_BYTES:
                fail('file_size', '下载文件大小无效或超过 512 MiB')
            received = 0
            with Path(target).open('wb') as stream:
                while True:
                    if time.monotonic() > deadline:
                        fail('download_timeout', '下载超时，请稍后重试')
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    received += len(block)
                    if received > MAX_BYTES:
                        fail('file_size', '下载超过 512 MiB，已停止')
                    stream.write(block)
                    progress('downloading', received, total)
            if total is not None and received != total:
                fail('incomplete_download', '下载不完整，请重试')
            return
        except (OSError, http.client.HTTPException, ValueError):
            fail('network_error', '下载失败，网络、TLS 或服务器响应异常')
        finally:
            connection.close()
    fail('unsafe_redirect', '下载重定向次数过多')


def extract_dll(archive, target, progress):
    """Extract one named DLL into our own file; never extract archive paths."""
    try:
        with zipfile.ZipFile(archive) as package:
            entries = package.infolist()
            if len(entries) > 128:
                fail('invalid_archive', 'ZIP 条目过多')
            candidates = []
            for entry in entries:
                path = PurePosixPath(entry.filename.replace('\\', '/'))
                if path.is_absolute() or '..' in path.parts or ':' in entry.filename:
                    fail('unsafe_archive', 'ZIP 包含不安全路径，已拒绝')
                if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                    fail('unsafe_archive', 'ZIP 不允许符号链接')
                if path.name.lower() == NAME and not entry.is_dir():
                    candidates.append(entry)
            if len(candidates) != 1:
                fail('invalid_archive', 'ZIP 必须只包含一个 nvngx_dlssnr.dll')
            item = candidates[0]
            if item.flag_bits & 1 or not 128 <= item.file_size <= MAX_BYTES:
                fail('invalid_archive', 'ZIP 内 DLL 加密或大小超限')
            size = 0
            with package.open(item) as reader, Path(target).open('wb') as writer:
                for block in iter(lambda: reader.read(1024 * 1024), b''):
                    size += len(block)
                    if size > MAX_BYTES:
                        fail('file_size', 'ZIP 内 DLL 超过 512 MiB')
                    writer.write(block)
                    progress('extracting', size, item.file_size)
            if size != item.file_size:
                fail('invalid_archive', 'ZIP 内 DLL 不完整')
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
        if isinstance(exc, RuntimeErrorDetail):
            raise
        fail('invalid_archive', 'ZIP 损坏、加密或采用不支持的压缩方式')


class RuntimeManager:
    def __init__(self, root=None, identity_checker=windows_identity, downloader=fetch_https):
        default = Path(os.environ.get('LUNARC_DATA_DIR') or
                       Path(os.environ.get('LOCALAPPDATA') or Path.home() / '.cache') / 'Lunarc')
        self.root = Path(root) if root else default / 'runtimes/dlssnr'
        self.config = self.root / 'registry.json'
        self.identity_checker = identity_checker
        self.downloader = downloader
        self.lock = threading.RLock()
        self.operation_lock = threading.Lock()
        self.operation = None
        self.verified = {}
        self.cancel_event = threading.Event()

    def load(self):
        with self.lock:
            if not self.config.exists():
                return {'schema': 1, 'active': None, 'versions': [], 'sources': []}
            try:
                data = json.loads(self.config.read_text(encoding='utf-8'))
                if (data['schema'] != 1 or not isinstance(data['versions'], list) or
                        not isinstance(data['sources'], list) or
                        not (data['active'] is None or isinstance(data['active'], str) and _SHA.fullmatch(data['active']))):
                    raise ValueError()
                for record in data['versions']:
                    if not isinstance(record, dict) or not _SHA.fullmatch(record.get('id', '')):
                        raise ValueError()
                for source in data['sources']:
                    if (not isinstance(source, dict) or source.get('kind') not in ('directory', 'download') or
                            not all(isinstance(source.get(k), str) for k in ('id', 'location', 'label', 'sha256')) or
                            source['kind'] == 'download' and not _SHA.fullmatch(source['sha256'])):
                        raise ValueError()
                    if source.get('archive_sha256') and not _SHA.fullmatch(source['archive_sha256']):
                        raise ValueError()
                return data
            except (ValueError, KeyError, TypeError, AssertionError):
                fail('cache_metadata', '运行库缓存记录损坏；不会回退加载其他 DLL')

    def save(self, data):
        with self.lock:
            self.root.mkdir(parents=True, exist_ok=True)
            temp = self.root / ('registry-' + uuid.uuid4().hex + '.tmp')
            try:
                temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
                os.replace(temp, self.config)
            finally:
                temp.unlink(missing_ok=True)

    def path(self, version):
        if not isinstance(version, str) or not _SHA.fullmatch(version):
            fail('invalid_id', '运行库版本标识无效')
        value = self.root / 'versions' / version / NAME
        if not value.resolve().is_relative_to(self.root.resolve()):
            fail('unsafe_cache', '运行库缓存路径不安全')
        return value

    def progress(self, phase, received=0, total=None):
        if phase != 'idle' and self.cancel_event.is_set():
            fail('cancelled', '已取消运行库导入或下载')
        with self.lock:
            self.operation = {'phase': phase, 'received': received, 'total': total}

    def cancel(self):
        self.cancel_event.set()

    def verified_path(self, version, force_signature=False):
        path = self.path(version)
        if not path.is_file():
            fail('cache_missing', '选定的 DLSSNR 缓存文件已丢失，请重新导入；不会静默换库')
        # Rehash before every load; signature results may be reused only for
        # identical bytes verified in this process, never trusted from registry.
        digest = sha256(path)
        if digest != version:
            fail('cache_tampered', '缓存 DLL 的 SHA-256 已改变，请重新导入')
        key = (str(path.resolve()), digest)
        with self.lock:
            known = self.verified.get(key)
        if force_signature or not known:
            known = inspect(path, version, self.identity_checker)
            with self.lock:
                self.verified[key] = known
        return str(path), known

    def status(self):
        data = self.load()
        error = None
        active = next((v for v in data['versions'] if v.get('id') == data['active']), None)
        if data['active']:
            try:
                path, actual = self.verified_path(data['active'])
                active = {**actual, 'path': path}
            except RuntimeErrorDetail as exc:
                error = {'code': exc.code, 'message': str(exc)}
        else:
            error = {'code': 'runtime_missing', 'message': '尚未选择 DLSSNR 运行库，请导入合法取得的 nvngx_dlssnr.dll'}
        catalog = [{**entry, 'id': digest, 'sha256': digest,
                    'downloaded': any(v['id'] == digest for v in data['versions'])}
                   for digest, entry in KNOWN_VERSIONS.items()]
        return {**data, 'active': active, 'error': error, 'cache_dir': str(self.root), 'catalog': catalog,
                'operation': self.operation,
                'legacy_path': str(Path(__file__).resolve().parent / NAME) if (Path(__file__).resolve().parent / NAME).is_file() else None}

    def add_source(self, kind, location, label, expected_sha='', confirmed=False,
                   archive_sha256=''):
        if not confirmed:
            fail('source_not_approved', '请明确确认来源与使用许可后添加')
        if kind not in ('directory', 'download') or not label.strip() or len(label) > 120:
            fail('invalid_source', '来源类型或名称无效')
        expected_sha = expected_sha.strip().lower()
        archive_sha256 = archive_sha256.strip().lower()
        if archive_sha256 and not _SHA.fullmatch(archive_sha256):
            fail('invalid_hash', 'ZIP SHA-256 必须是 64 位十六进制值')
        if expected_sha and not _SHA.fullmatch(expected_sha):
            fail('invalid_hash', 'SHA-256 必须是 64 位十六进制值')
        if kind == 'download':
            parsed_url(location)
            if not expected_sha:
                fail('hash_required', '下载来源必须提供预期 SHA-256')
        else:
            location = str(Path(location).resolve())
            if not Path(location).is_dir():
                fail('directory_missing', '可信来源目录不存在')
        with self.lock:
            data = self.load()
            if len(data['sources']) >= 32:
                fail('source_limit', '最多保存 32 个来源')
            entry = {'id': uuid.uuid4().hex, 'kind': kind, 'location': location,
                     'label': label.strip(), 'sha256': expected_sha, 'archive_sha256': archive_sha256}
            data['sources'].append(entry)
            self.save(data)
        return entry

    def remove_source(self, source_id):
        with self.lock:
            data = self.load()
            data['sources'] = [s for s in data['sources'] if s.get('id') != source_id]
            self.save(data)

    def import_file(self, path=None, expected_sha='', source_id=None, activate=False,
                    catalog_id=None, confirmed=False):
        if not self.operation_lock.acquire(blocking=False):
            fail('operation_busy', '已有运行库操作进行中，请稍候')
        temp = None
        archive_temp = None
        self.cancel_event.clear()
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            temp = self.root / ('incoming-' + uuid.uuid4().hex + '.dll')
            origin = {'kind': 'manual'}
            if catalog_id:
                if not confirmed:
                    fail('source_not_approved', '请确认社区来源与使用许可后下载')
                known = KNOWN_VERSIONS.get(catalog_id)
                if not known:
                    fail('version_missing', '版本不在已核对的目录中')
                source = dict(kind='download', label=known['source'], location=known['url'],
                              sha256=catalog_id, archive_sha256=known['archive_sha256'])
            elif source_id:
                source = next((s for s in self.load()['sources'] if s.get('id') == source_id), None)
                if not source:
                    fail('source_missing', '来源已删除，请重新选择')
            else:
                source = None
            if source:
                origin = {'kind': source['kind'], 'label': source['label']}
                expected_sha = source['sha256']
                if source['kind'] == 'download':
                    self.progress('downloading')
                    archive_hash = source.get('archive_sha256')
                    if archive_hash:
                        download = self.root / 'downloads' / (archive_hash + '.zip')
                        download.parent.mkdir(exist_ok=True)
                        if download.exists() and sha256(download) != archive_hash:
                            fail('cache_tampered', '下载缓存哈希已改变，请移除损坏的 ZIP 后重试')
                        if not download.exists():
                            archive_temp = self.root / ('download-' + uuid.uuid4().hex + '.part')
                            self.downloader(source['location'], archive_temp, self.progress)
                            if sha256(archive_temp) != archive_hash:
                                fail('hash_mismatch', 'ZIP SHA-256 不匹配，已拒绝')
                            os.replace(archive_temp, download)
                        extract_dll(download, temp, self.progress)
                    else:
                        self.downloader(source['location'], temp, self.progress)
                else:
                    path = str(Path(source['location']) / NAME)
            if not temp.exists() and path and Path(path).suffix.lower() == '.zip':
                if not Path(path).is_file() or Path(path).stat().st_size > MAX_BYTES:
                    fail('file_size', 'ZIP 文件缺失或超过 512 MiB')
                extract_dll(path, temp, self.progress)
            if not temp.exists():
                if not path or not Path(path).is_file():
                    fail('file_missing', '未找到要导入的 DLSSNR DLL')
                if not 128 <= Path(path).stat().st_size <= MAX_BYTES:
                    fail('file_size', 'DLL 大小无效或超过 512 MiB')
                self.progress('copying')
                with Path(path).open('rb') as reader, temp.open('wb') as writer:
                    size = 0
                    for block in iter(lambda: reader.read(4 * 1024 * 1024), b''):
                        size += len(block)
                        if size > MAX_BYTES:
                            fail('file_size', '导入文件在复制期间增长，已停止')
                        writer.write(block)
                        self.progress('copying', size, Path(path).stat().st_size)
            expected_sha = expected_sha.strip().lower()
            if expected_sha and not _SHA.fullmatch(expected_sha):
                fail('invalid_hash', 'SHA-256 必须是 64 位十六进制值')
            self.progress('verifying')
            record = inspect(temp, expected_sha or None, self.identity_checker)
            self.progress('verifying')
            destination = self.path(record['id'])
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                # Re-import may repair a corrupt cache, but no running session
                # can be switched by replacing bytes at an existing path.
                if sha256(destination) != record['id']:
                    fail('cache_tampered', '同版本缓存已损坏，请先删除该版本后重新导入')
            else:
                os.replace(temp, destination)
            with self.lock:
                data = self.load()
                data['versions'] = [v for v in data['versions'] if v.get('id') != record['id']]
                data['versions'].append({**record, 'origin': origin, 'imported_at': time.time()})
                if activate:
                    data['active'] = record['id']
                self.save(data)
                self.verified[(str(destination.resolve()), record['id'])] = record
            return {**record, 'path': str(destination)}
        except RuntimeErrorDetail:
            raise
        except OSError:
            fail('cache_io', '运行库文件操作失败，请检查文件占用、缓存目录权限和磁盘空间')
        finally:
            try:
                if temp:
                    temp.unlink(missing_ok=True)
                if archive_temp:
                    archive_temp.unlink(missing_ok=True)
            except OSError:
                pass  # Never use an incomplete incoming file; retain the original error.
            finally:
                self.progress('idle')
                self.operation_lock.release()

    def activate(self, version, allow_unrecognized=False):
        # Keep the old argument for API compatibility. Only file structure and
        # integrity block selection; an unrecognized identity is informational.
        if not self.operation_lock.acquire(blocking=False):
            fail('operation_busy', '运行库操作进行中')
        try:
            if version is not None:
                self.verified_path(version)
            with self.lock:
                data = self.load()
                if version is not None and not any(v.get('id') == version for v in data['versions']):
                    fail('version_missing', '缓存中没有该版本记录')
                data['active'] = version
                self.save(data)
        finally:
            self.operation_lock.release()

    def delete(self, version):
        if not self.operation_lock.acquire(blocking=False):
            fail('operation_busy', '运行库操作进行中')
        try:
            with self.lock:
                data = self.load()
                if data['active'] == version:
                    fail('active_version', '不能删除选定版本，请先切换或取消选择')
                path = self.path(version)
                try:
                    path.unlink(missing_ok=True)
                    if path.parent.exists():
                        path.parent.rmdir()
                except OSError:
                    fail('file_in_use', 'DLL 正在使用或目录不可写，请关闭会话后重试')
                data['versions'] = [v for v in data['versions'] if v.get('id') != version]
                self.save(data)
        finally:
            self.operation_lock.release()


_manager = None
_manager_root = None
_manager_lock = threading.Lock()


def manager():
    global _manager, _manager_root
    # Tests and environment workers may change the user-data root explicitly.
    key = (os.environ.get('LUNARC_DATA_DIR'), os.environ.get('LOCALAPPDATA'))
    with _manager_lock:
        if _manager is None or key != _manager_root:
            _manager, _manager_root = RuntimeManager(), key
        return _manager


def selection_key():
    value = manager()
    active = value.load()['active']
    if not active:
        return (str(value.config), None)
    try:
        stat = value.path(active).stat()
        return (str(value.config), active, stat.st_mtime_ns, stat.st_size)
    except FileNotFoundError:
        return (str(value.config), active, 'missing')


def resolve_runtime():
    value = manager()
    active = value.load()['active']
    if active:
        return value.verified_path(active)[0]
    override = os.environ.get('LUNARC_DLSSNR_DLL')
    # Development-only migration of the old user-supplied file. A released app
    # never loads a root copy or downloads a runtime without user selection.
    legacy = Path(__file__).resolve().parent / NAME
    if not value.config.exists() and (override or (os.environ.get('LUNARC_STANDALONE') != '1' and legacy.is_file())):
        return value.import_file(override or str(legacy), activate=True)['path']
    fail('runtime_missing', '缺少 DLSSNR 运行库。请在“运行与诊断 → DLSSNR 运行库”导入合法取得的 nvngx_dlssnr.dll；软件不会自动下载未知来源')
