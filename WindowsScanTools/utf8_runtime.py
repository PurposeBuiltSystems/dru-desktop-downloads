"""Enable the documented Windows UTF-8 process code page for native file APIs."""
from pathlib import Path
import os
import subprocess
import xml.etree.ElementTree as ET


def configure(interpreter, working_directory):
    kits = Path(os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')) / 'Windows Kits/10/bin'
    tools = sorted(kits.glob('*/x64/mt.exe'))
    if not tools:
        raise RuntimeError('The Microsoft manifest tool is missing')
    tool = tools[-1]
    manifest = working_directory / 'python-runtime.manifest'
    subprocess.run([str(tool), '-nologo', '-inputresource:' + str(interpreter) + ';#1', '-out:' + str(manifest)], check=True)
    ET.register_namespace('', 'urn:schemas-microsoft-com:asm.v1')
    ET.register_namespace('asmv3', 'urn:schemas-microsoft-com:asm.v3')
    tree = ET.parse(manifest)
    root = tree.getroot()
    namespace = '{urn:schemas-microsoft-com:asm.v3}'
    application = root.find(namespace + 'application')
    if application is None:
        application = ET.SubElement(root, namespace + 'application')
    settings = application.find(namespace + 'windowsSettings')
    if settings is None:
        settings = ET.SubElement(application, namespace + 'windowsSettings')
    tag = '{http://schemas.microsoft.com/SMI/2019/WindowsSettings}activeCodePage'
    code_page = settings.find(tag)
    if code_page is None:
        code_page = ET.SubElement(settings, tag)
    code_page.text = 'UTF-8'
    tree.write(manifest, encoding='utf-8', xml_declaration=True)
    subprocess.run([str(tool), '-nologo', '-manifest', str(manifest), '-outputresource:' + str(interpreter) + ';#1'], check=True)
    interpreter.with_name('PURPOSEBUILT-RUNTIME-NOTICE.txt').write_text(
        'PurposeBuilt modifies the Python launcher manifest to enable UTF-8 filesystem paths in native reconstruction libraries. Python license and attribution accompany this runtime in LICENSE.txt.\n', encoding='utf-8')
