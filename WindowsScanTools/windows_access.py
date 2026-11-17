"""Windows DPAPI-protected device identity and the shared signed-license protocol."""
import base64
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption
from platform_runtime import lock_file


def b64(data):
    return base64.b64encode(data).decode('ascii')


def unb64(value):
    return base64.b64decode(value, validate=True)


def protect(data, decrypt=False):
    class Blob(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_ubyte))]
    buffer = ctypes.create_string_buffer(data)
    incoming = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    outgoing = Blob()
    api = ctypes.WinDLL('crypt32', use_last_error=True)
    operation = api.CryptUnprotectData if decrypt else api.CryptProtectData
    operation.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    operation.restype = wintypes.BOOL
    if not operation(ctypes.byref(incoming), None, None, None, None, 1, ctypes.byref(outgoing)):
        raise OSError('The Windows account could not access the desktop license.')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    try:
        return ctypes.string_at(outgoing.data, outgoing.length)
    finally:
        kernel.LocalFree(outgoing.data)


@contextmanager
def state_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a+') as stream:
        deadline = time.monotonic() + 10
        while True:
            try:
                lock_file(stream)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Another license check is still running. Please retry.')
                time.sleep(.1)
        yield


def read_state(path):
    path = Path(path)
    if path.exists():
        return json.loads(protect(path.read_bytes(), decrypt=True))
    private = Ed25519PrivateKey.generate().private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    return {'privateKey': b64(private), 'lastSeen': time.time()}


def save_state(path, record):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + secrets.token_hex(8) + '.tmp')
    try:
        temporary.write_bytes(protect(json.dumps(record).encode()))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify(envelope, configuration, device, now=None):
    now = time.time() if now is None else now
    payload = unb64(envelope['payload'])
    Ed25519PublicKey.from_public_bytes(unb64(configuration['public_key'])).verify(unb64(envelope['signature']), payload)
    claims = json.loads(payload)
    if (claims['product'] != configuration['product'] or claims['major_version'] != configuration['major_version']
            or claims['issued_at'] > now + 300):
        raise ValueError('This license does not match this app version or the computer clock.')
    if claims['kind'] == 'perpetual':
        if claims.get('expires_at') is not None:
            raise ValueError('Invalid desktop license expiry.')
    elif (claims['kind'] not in ['trial', 'pro'] or claims.get('device_public_key') != b64(device)
          or not claims.get('expires_at', 0) > max(now, claims['issued_at'])):
        raise ValueError('The trial or subscription has expired. Refresh your license to continue.')
    return claims


def status(configuration_path, state_path):
    configuration = json.loads(Path(configuration_path).read_text())
    with state_lock(state_path):
        record = read_state(state_path)
        now = time.time()
        if now + 300 < record['lastSeen']:
            return {'allowed': False, 'kind': 'clock', 'message': 'Correct the Windows date and time before processing.'}
        record['lastSeen'] = max(record['lastSeen'], now)
        save_state(state_path, record)
        if not record.get('license'):
            return {'allowed': False, 'kind': 'unlicensed', 'message': 'Start a 14-day trial or activate your desktop license in Account.'}
        device = Ed25519PrivateKey.from_private_bytes(unb64(record['privateKey'])).public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        try:
            claims = verify(record['license'], configuration, device, now)
        except Exception:
            return {'allowed': False, 'kind': 'expired', 'message': 'The license could not be verified. Open Account to refresh or import it.'}
        days = math.ceil((claims['expires_at'] - now) / 86400) if claims.get('expires_at') else None
        message = 'Desktop license active for major version 1.'
        if claims['kind'] == 'trial':
            message = f'{days} days remaining in your trial.'
        if claims['kind'] == 'pro':
            message = 'Complimentary DRU Pro access.' if claims.get('source') == 'complimentary' else f'DRU Pro access active for {days} more days before renewal.'
        return {'allowed': True, 'kind': claims['kind'], 'message': message, 'days_remaining': days}


class ServiceError(ValueError):
    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


def post(url, body, headers=None):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read(65537)
            if len(data) > 65536:
                raise ValueError('The license service returned an oversized response.')
            return json.loads(data)
    except urllib.error.HTTPError as error:
        try:
            code = json.loads(error.read(65536)).get('error', '')
        except (ValueError, OSError):
            code = ''
        messages = {401: 'Sign in again to continue.', 402: 'This account needs active DRU Pro access.',
                    403: 'The license is unavailable. Refresh or contact PurposeBuilt support.',
                    409: 'The request was already handled. Refresh your license before retrying.'}
        raise ServiceError(messages.get(error.code, 'The license service is temporarily unavailable. Please retry.'), code) from None


def perform(action, configuration_path, state_path, auth_token=None):
    configuration = json.loads(Path(configuration_path).read_text())
    with state_lock(state_path):
        record = read_state(state_path)
        save_state(state_path, record)
        private = Ed25519PrivateKey.from_private_bytes(unb64(record['privateKey']))
        device = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        nonce = post(configuration['origin'] + '/challenge', {'public_key': b64(device)})['nonce']
        payload = {'public_key': b64(device), 'nonce': nonce, 'action': action}
        if record.get('license'):
            payload['license'] = record['license']
        if auth_token:
            payload['auth_token'] = auth_token
        raw = json.dumps(payload, separators=(',', ':'), sort_keys=True).encode()
        try:
            response = post(configuration['origin'] + '/' + action, {'payload': b64(raw), 'signature': b64(private.sign(raw))})
        except ServiceError as error:
            if error.code == 'license_revoked':
                record.pop('license', None)
                save_state(state_path, record)
            raise
        if response.get('license'):
            verify(response['license'], configuration, device)
            record['license'] = response['license']
            record['lastSeen'] = max(record['lastSeen'], time.time())
            save_state(state_path, record)
        return response


def transfer(action, configuration_path, state_path, file):
    configuration = json.loads(Path(configuration_path).read_text())
    with state_lock(state_path):
        record = read_state(state_path)
        if action == 'export':
            if not record.get('license'):
                raise ValueError('Activate a desktop license before saving a license file.')
            Path(file).write_text(json.dumps(record['license'], indent=2), encoding='utf-8')
            return
        data = Path(file).read_bytes()
        if len(data) > 65536:
            raise ValueError('The license file exceeds the size limit.')
        envelope = json.loads(data)
        device = Ed25519PrivateKey.from_private_bytes(unb64(record['privateKey'])).public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        verify(envelope, configuration, device)
        record['license'] = envelope
        record['lastSeen'] = max(record['lastSeen'], time.time())
        save_state(state_path, record)


def sign_in(provider, configuration_path):
    configuration = json.loads(Path(configuration_path).read_text())
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip('=')
    callback = '/auth/callback'
    captured = {}
    class Receiver(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            parsed = urllib.parse.urlsplit(self.path)
            values = urllib.parse.parse_qs(parsed.query)
            if parsed.path != callback or 'code' not in values or len(self.path) > 16384:
                self.send_error(400, 'Sign-in was not completed.')
                return
            captured['code'] = values['code'][0]
            message = b'<!doctype html><title>PurposeBuilt Scan Tools</title><p>Return to PurposeBuilt Scan Tools to finish activation.</p>'
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(message)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(message)
    with HTTPServer(('127.0.0.1', 8766), Receiver) as server:
        server.timeout = 1
        redirect = 'http://127.0.0.1:' + str(server.server_port) + callback
        parameters = urllib.parse.urlencode({'provider': provider, 'redirect_to': redirect, 'code_challenge': challenge, 'code_challenge_method': 's256'})
        webbrowser.open(configuration['auth_url'] + '/authorize?' + parameters)
        deadline = time.monotonic() + 300
        while not captured and time.monotonic() < deadline:
            server.handle_request()
        if not captured:
            raise TimeoutError('Sign-in timed out. Start sign-in again from Account.')
    response = post(configuration['auth_url'] + '/token?grant_type=pkce', {'auth_code': captured['code'], 'code_verifier': verifier}, {'apikey': configuration['publishable_key']})
    return response['access_token']
