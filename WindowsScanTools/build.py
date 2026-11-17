"""Build a self-contained, per-user Windows x64 installer from the tested engine."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUNDLE = HERE / 'bundle'


def main():
    assert sys.platform == 'win32'
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir()
    python = BUNDLE / 'python'
    shutil.copytree(Path(sys.base_prefix), python, ignore=shutil.ignore_patterns('site-packages', '__pycache__', 'Scripts', 'Doc', 'Tools'))
    from utf8_runtime import configure
    configure(python / 'python.exe', HERE)
    site = python / 'Lib/site-packages'
    site.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-compile', '--target', str(site),
                    'torch==2.11.0+cpu', 'torchvision==0.26.0+cpu', 'torchaudio==2.11.0+cpu',
                    '-r', str(HERE / 'requirements.txt'), '--extra-index-url', 'https://download.pytorch.org/whl/cpu'], check=True)
    resources = BUNDLE / 'resources'
    worker = resources / 'worker'
    worker.mkdir(parents=True)
    for name in ['worker','reconstruct','fast_alignment','dense_photo','quantized_weights','photo_stages',
                 'stereo_photo','fuse_photo','repair_room','repair_requests','repair_suggestions','result_storage',
                 'scan_branding','scan_identity','desktop_access','platform_runtime','scan_image_io']:
        shutil.copy2(ROOT / 'MacScanWorker' / (name + '.py'), worker / (name + '.py'))
    shutil.copy2(ROOT / 'MacScanWorker/desktop-access.json', worker / 'desktop-access.json')
    shutil.copy2(ROOT / 'MacScanWorker/repair_editor.html', worker / 'repair_editor.html')
    shutil.copytree(ROOT / 'MacScanWorker/Branding', worker / 'Branding')
    for name in ['windows_access.py','windows_entry.py']:
        shutil.copy2(HERE / name, resources / 'worker' / name)
    shutil.copytree(ROOT / 'WindowsShared/scan_recon', resources / 'scan_recon')
    shutil.copytree(ROOT / 'WindowsShared/vendor', resources / 'vendor')
    artifact = json.loads((HERE / 'model-artifact.json').read_text())
    model_zip = HERE / 'model.zip'
    urllib.request.urlretrieve(artifact['url'], model_zip)
    with model_zip.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == artifact['sha256'], 'Model checksum mismatch'
    with zipfile.ZipFile(model_zip) as archive:
        for entry in archive.infolist():
            assert Path(entry.filename).parts[0] == 'map-anything-int4-g32' and len(Path(entry.filename).parts) == 2
        archive.extractall(resources / 'models')
    # App-local Microsoft CRT files remove a separate administrator-level runtime installation.
    vswhere = Path(os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')) / 'Microsoft Visual Studio/Installer/vswhere.exe'
    studio = Path(subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-property', 'installationPath'], text=True).strip())
    crt_folders = sorted(studio.glob('VC/Redist/MSVC/*/x64/Microsoft.VC*.CRT'))
    if not crt_folders:
        raise RuntimeError('Microsoft redistributable CRT files are missing')
    for file in crt_folders[-1].glob('*.dll'):
        shutil.copy2(file, python / file.name)
    subprocess.run([str(python / 'python.exe'), '-I', '-c',
                    'from PIL import Image; import sys; Image.open(sys.argv[1]).save(sys.argv[2], sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])',
                    str(resources / 'worker/Branding/dru-logo.png'), str(HERE / 'app.ico')], check=True)
    compiler = Path(os.environ['WINDIR']) / 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
    subprocess.run([str(compiler), '/nologo', '/target:winexe', '/platform:x64', '/optimize+', '/win32icon:' + str(HERE / 'app.ico'), '/reference:System.Windows.Forms.dll', '/reference:System.Drawing.dll', '/reference:System.Web.Extensions.dll', '/out:' + str(BUNDLE / 'PurposeBuiltScanTools.exe'), str(HERE / 'Launcher.cs')], check=True)
    # Resolve imports using the relocated runtime before starting a long reconstruction test.
    subprocess.run([str(python / 'python.exe'), '-I', '-X', 'utf8', str(resources / 'worker/windows_entry.py'), 'self-test', '--support-root', str(HERE / 'QA support'), '--scan-root', str(HERE / 'QA scans')], check=True)
    inno = Path(os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')) / 'Inno Setup 6/ISCC.exe'
    subprocess.run([str(inno),str(HERE / 'installer.iss')],check=True,cwd=HERE)
    installer = HERE / 'output/PurposeBuilt-Scan-Tools-Windows-x64.exe'
    with installer.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    (HERE / 'output/installer-manifest.json').write_text(json.dumps({'name':'PurposeBuilt Scan Tools','version':'1.0','architecture':'Windows x64','sha256':digest,'bytes':installer.stat().st_size,'signature':'unsigned','model_storage':'four-bit','source_revision':os.environ.get('GITHUB_SHA')},indent=2),encoding='utf-8')
    print('Installer created', installer.stat().st_size, digest)


if __name__ == '__main__':
    main()
