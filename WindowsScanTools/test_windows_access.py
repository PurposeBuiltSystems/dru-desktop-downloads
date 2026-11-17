"""Signature, revocation and browser callback regressions for the Windows client."""
import base64
import hashlib
from http.server import HTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'MacScanWorker')]
import windows_access as access
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption


class WindowsLicenseTests(unittest.TestCase):
    def setUp(self):
        self.issuer = Ed25519PrivateKey.generate()
        self.device = Ed25519PrivateKey.generate()
        self.public = self.device.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        self.config = {'public_key':access.b64(self.issuer.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)),
            'product':'dru_desktop_scan_tools_v1','major_version':1,'origin':'https://license.example.invalid',
            'auth_url':'https://auth.example.invalid','publishable_key':'public-test-only'}

    def envelope(self, **updates):
        claims = {'product':self.config['product'],'major_version':1,'kind':'trial',
                  'issued_at':int(time.time())-60,'expires_at':int(time.time())+3600,'device_public_key':access.b64(self.public)}
        claims.update(updates)
        raw = json.dumps(claims).encode()
        return {'payload':access.b64(raw),'signature':access.b64(self.issuer.sign(raw))}

    def test_signature_expiry_product_and_device_are_checked(self):
        self.assertEqual(access.verify(self.envelope(),self.config,self.public)['kind'],'trial')
        for changes in [{'product':'other_product'},{'major_version':2},{'expires_at':0},{'device_public_key':'wrong'},{'issued_at':int(time.time())+1000}]:
            with self.assertRaises(ValueError):access.verify(self.envelope(**changes),self.config,self.public)
        envelope=self.envelope();envelope['payload']=access.b64(b'{}')
        with self.assertRaises(Exception):access.verify(envelope,self.config,self.public)

    def test_perpetual_license_keeps_existing_transfer_behavior(self):
        self.assertEqual(access.verify(self.envelope(kind='perpetual',expires_at=None),self.config,b'another-device')['kind'],'perpetual')
        with self.assertRaises(ValueError):access.verify(self.envelope(kind='perpetual'),self.config,self.public)

    def test_verified_server_revocation_clears_cached_license(self):
        with tempfile.TemporaryDirectory() as directory:
            config=Path(directory)/'config.json';config.write_text(json.dumps(self.config))
            record={'privateKey':access.b64(self.device.private_bytes(Encoding.Raw,PrivateFormat.Raw,NoEncryption())),
                    'lastSeen':time.time(),'license':self.envelope()}
            with patch.object(access,'read_state',return_value=record), patch.object(access,'save_state') as save, patch.object(access,'post',side_effect=[{'nonce':'fixture'},access.ServiceError('Revoked','license_revoked')]):
                with self.assertRaises(access.ServiceError):access.perform('license',config,Path(directory)/'state.dpapi')
                self.assertNotIn('license',save.call_args.args[1])

    def test_google_pkce_browser_return_is_bound_to_its_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            config=Path(directory)/'config.json';config.write_text(json.dumps(self.config))
            captured={};threads=[]
            def browser(url):
                values=urllib.parse.parse_qs(urllib.parse.urlsplit(url).query);captured.update(values)
                def callback():
                    with urllib.request.urlopen(values['redirect_to'][0]+'?code=fixture-code',timeout=5) as response:self.assertEqual(response.status,200)
                thread=threading.Thread(target=callback);thread.start();threads.append(thread);return True
            with patch.object(access,'HTTPServer',side_effect=lambda address,handler:HTTPServer(('127.0.0.1',0),handler)), patch.object(access.webbrowser,'open',side_effect=browser), patch.object(access,'post',return_value={'access_token':'fixture-token'}) as post:
                self.assertEqual(access.sign_in('google',config),'fixture-token')
                request=post.call_args.args[1]
                challenge=base64.urlsafe_b64encode(hashlib.sha256(request['code_verifier'].encode()).digest()).decode().rstrip('=')
                self.assertEqual(captured['code_challenge'][0],challenge)
                self.assertEqual(request['auth_code'],'fixture-code')
            for thread in threads:thread.join(5)


if __name__=='__main__':unittest.main()
