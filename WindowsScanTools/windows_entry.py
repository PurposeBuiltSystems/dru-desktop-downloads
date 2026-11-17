"""Trusted entry points for the installed Windows launcher and local worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from uuid import uuid4

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def configure(support, scans, resources):
    support.mkdir(parents=True, exist_ok=True)
    scans.mkdir(parents=True, exist_ok=True)
    for name in ['Inbox', 'Processing', 'Results', 'Archive', 'Failed', 'Repairs']:
        (scans / name).mkdir(exist_ok=True)
    config = {'scan_root': str(scans), 'dru_scripts': str(resources / 'scan_recon'),
              'mapanything_source': str(resources / 'vendor/map-anything'), 'native_library_path': '',
              'quantized_model': str(resources / 'models/map-anything-int4-g32'), 'viewer_port': 8765,
              'fast_alignment': True, 'pose_guided_matching': False, 'dense_max_views': 44,
              'dense_long_side': 392, 'dense_chunk': 4, 'native_int4': False, 'cleanup_after_success': True,
              'access_platform': 'windows', 'access_configuration': str(resources / 'worker/desktop-access.json'),
              'access_state': str(support / 'desktop-access.dpapi')}
    path = support / 'config.json'
    temporary = support / ('config-' + uuid4().hex + '.tmp')
    serialized = json.dumps(config, indent=2)
    if not path.exists() or path.read_text(encoding='utf-8') != serialized:
        temporary.write_text(serialized, encoding='utf-8')
        os.replace(temporary, path)
    return config, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['bootstrap', 'worker', 'access', 'import-scan', 'self-test'])
    parser.add_argument('--support-root', type=Path, required=True)
    parser.add_argument('--scan-root', type=Path, required=True)
    parser.add_argument('--action', choices=['status', 'trial', 'license', 'checkout', 'google', 'apple', 'import', 'export'])
    parser.add_argument('--file', type=Path)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    resources = HERE.parent
    support, scans = args.support_root.resolve(), args.scan_root.resolve()
    config, config_path = configure(support, scans, resources)
    os.environ.update(PYTHONUTF8='1', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', TORCH_HOME=str(resources / 'vendor/torch-cache'))
    os.environ['PYTHONPATH'] = os.pathsep.join([str(HERE), config['dru_scripts'], config['mapanything_source']])
    if args.command == 'bootstrap':
        print(json.dumps({'config': str(config_path), 'scans': str(scans)}))
    elif args.command == 'worker':
        from worker import main as worker_main
        sys.argv = ['worker.py', '--config', str(config_path)] + (['--once'] if args.once else [])
        worker_main()
    elif args.command == 'access':
        from windows_access import perform, status, transfer, sign_in
        if args.action in ['google', 'apple']:
            perform('pro', config['access_configuration'], config['access_state'], sign_in(args.action, config['access_configuration']))
        elif args.action in ['import', 'export']:
            if args.file is None:
                raise ValueError('Choose a license file.')
            transfer(args.action, config['access_configuration'], config['access_state'], args.file)
        elif args.action in ['trial', 'license', 'checkout']:
            response = perform(args.action, config['access_configuration'], config['access_state'])
            if args.action == 'checkout':
                from urllib.parse import urlsplit
                import webbrowser
                url = urlsplit(response['url'])
                if url.scheme != 'https' or url.hostname != 'checkout.stripe.com':
                    raise ValueError('The payment service returned an invalid address.')
                webbrowser.open(response['url'])
        print(json.dumps(status(config['access_configuration'], config['access_state'])))
    elif args.command == 'import-scan':
        if args.file is None or not args.file.is_file():
            raise ValueError('Choose an exported photo scan ZIP.')
        from worker import validate_source
        validate_source(args.file)
        destination = scans / 'Inbox' / (args.file.stem[:100] + '-' + uuid4().hex[:8] + '.zip')
        temporary = destination.with_suffix('.partial')
        try:
            shutil.copyfile(args.file, temporary)
            # A copy protects the user's original if the companion is closed during capture processing.
            with args.file.open('rb') as original, temporary.open('rb') as copied:
                if hashlib.file_digest(original, 'sha256').digest() != hashlib.file_digest(copied, 'sha256').digest():
                    raise ValueError('The imported ZIP failed its integrity check.')
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        print(json.dumps({'imported': destination.name}))
    else:
        sys.path.extend([config['dru_scripts'], config['mapanything_source']])
        import torch
        import pycolmap
        import open3d
        import cv2
        import package_deliverable
        from windows_access import protect, status
        assert protect(protect(b'Windows account test'), decrypt=True) == b'Windows account test'
        print(json.dumps({'status': 'passed', 'python': sys.version.split()[0], 'torch': torch.__version__,
                          'pycolmap': pycolmap.__version__, 'open3d': open3d.__version__, 'dpapi': 'passed',
                          'license_state': status(config['access_configuration'], config['access_state'])['kind']}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'error': str(error) or 'The operation could not complete. Please retry.'}))
        raise SystemExit(1)
