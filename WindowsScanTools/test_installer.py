"""Run installer, relocated runtime, real trial and reconstruction checks on Windows."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

HERE = Path(__file__).resolve().parent


def run(arguments, **kwargs):
    result = subprocess.run([str(value) for value in arguments], check=True, capture_output=True, text=True, encoding='utf-8', **kwargs)
    print(result.stdout[-4000:])
    return result.stdout


def main():
    assert sys.platform == 'win32'
    installer = HERE / 'output/PurposeBuilt-Scan-Tools-Windows-x64.exe'
    with tempfile.TemporaryDirectory(prefix='PurposeBuilt installer ') as work:
        root = Path(work)
        installed = root / 'Installed App'
        support, scans = root / 'QA support', root / 'Project scans é-桥'
        run([installer, '/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART','/SP-','/DIR='+str(installed)])
        python = installed / 'python/python.exe'
        entry = installed / 'resources/worker/windows_entry.py'
        base = [python, '-I', '-X', 'utf8', entry]
        locations = ['--support-root',support,'--scan-root',scans]
        native = subprocess.run([str(installed / 'PurposeBuiltScanTools.exe'),'--self-test'],capture_output=True,text=True,encoding='utf-8',timeout=120)
        assert native.returncode == 0, native.stdout + native.stderr
        smoke = json.loads(run(base + ['self-test'] + locations).strip().splitlines()[-1])
        assert smoke['dpapi'] == 'passed'
        trial = json.loads(run(base + ['access','--action','trial'] + locations))
        assert trial['allowed'] and trial['kind'] == 'trial'
        state = support / 'desktop-access.dpapi'
        assert b'privateKey' not in state.read_bytes()
        # Reinstalling into the same directory keeps customer scan and license data.
        sentinel = scans / 'Archive/customer-data-preservation.txt'
        sentinel.write_text('Retain customer scans', encoding='utf-8')
        before = state.read_bytes()
        run([installer, '/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART','/SP-','/DIR='+str(installed)])
        assert state.read_bytes() == before and sentinel.exists()
        license_file = root / 'license.json'
        run(base + ['access','--action','export','--file',license_file] + locations)
        run(base + ['access','--action','import','--file',license_file] + locations)
        capture = HERE / 'synthetic-photo-scan.zip'
        run(base + ['import-scan','--file',capture] + locations)
        started = time.monotonic()
        result = subprocess.run([str(value) for value in base + ['worker','--once'] + locations],capture_output=True,text=True,encoding='utf-8',timeout=900)
        if result.returncode:
            for log in (scans / 'Results').rglob('*.log'):
                print(log.name, log.read_text(encoding='utf-8',errors='replace')[-6000:])
            raise RuntimeError(result.stdout[-3000:] + result.stderr[-3000:])
        statuses = [json.loads(path.read_text(encoding='utf-8')) for path in (scans / 'Results').glob('*/status.json')]
        assert len(statuses) == 1 and statuses[0]['state'] == 'done'
        status = statuses[0]
        output = scans / 'Results' / status['job_id']
        with zipfile.ZipFile(output / 'model-deliverable.zip') as archive:
            assert archive.testzip() is None
            assert 'BridgeCheck AR Scan Tools' in archive.read('model.html').decode()
            assert {'model-metres.ply','model-metres.obj','README.txt'}.issubset(archive.namelist())
        original = Path(status['archived_source'])
        assert hashlib.sha256(original.read_bytes()).digest() == hashlib.sha256(capture.read_bytes()).digest()
        with zipfile.ZipFile(original) as archive:
            manifest = json.loads(archive.read('manifest.json'))
        cameras = manifest['images']
        # Verify Unicode source image handling without relying on a particular username.
        image = root / 'permit-é-桥.jpg'
        with zipfile.ZipFile(original) as archive:
            image.write_bytes(archive.read(cameras[0]['filename']))
        env = os.environ.copy()
        env['PYTHONPATH'] = str(installed / 'resources/worker')
        run([python,'-P','-c','from scan_image_io import read_image; import sys; assert read_image(sys.argv[1]) is not None',image],env=env)
        report = {'state':'passed','platform':'Windows Server 2025 x64 GitHub-hosted runner',
                  'native_launcher_self_test':'passed','per_user_install':'passed','runtime_relocation':'passed',
                  'dpapi':'passed','real_trial_activation':'passed','license_import_export':'passed',
                  'reinstall_preserves_data':'passed','source_archive_integrity':'passed','unicode_image_path':'passed',
                  'synthetic_photos':len(cameras),'reconstruction_seconds':round(time.monotonic()-started,3),
                  'model_zip_bytes':(output / 'model-deliverable.zip').stat().st_size,'geometry_accuracy':'independent physical accuracy unverified',
                  'authenticode':'unsigned','interactive_google_apple_sign_in':'manual Windows verification pending'}
        run([installed / 'unins000.exe','/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART'])
        assert sentinel.exists() and state.exists() and original.exists()
        report['uninstall_preserves_customer_data'] = 'passed'
        (HERE / 'output/windows-verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
