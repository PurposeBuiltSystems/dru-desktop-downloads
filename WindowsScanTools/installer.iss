#define AppName "PurposeBuilt Scan Tools"
[Setup]
AppId={{694D11F4-2DAA-457C-9B02-A6F7A622CA91}
AppName={#AppName}
AppVersion=1.0
AppPublisher=PurposeBuilt Systems, LLC
AppPublisherURL=https://purposebuilt.systems
AppSupportURL=https://purposebuilt.systems
DefaultDirName={localappdata}\Programs\PurposeBuilt Scan Tools
DefaultGroupName={#AppName}
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.19041
DisableProgramGroupPage=yes
WizardStyle=modern
UninstallDisplayIcon={app}\PurposeBuiltScanTools.exe
OutputDir=output
OutputBaseFilename=PurposeBuilt-Scan-Tools-Windows-x64
Compression=lzma2/fast
SolidCompression=yes
LZMANumBlockThreads=2
CloseApplications=yes
RestartApplications=no
SetupLogging=yes
SetupIconFile=app.ico

[Files]
Source: "bundle\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\PurposeBuiltScanTools.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\PurposeBuiltScanTools.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts"; Flags: unchecked

[Run]
Filename: "{app}\PurposeBuiltScanTools.exe"; Description: "Open PurposeBuilt Scan Tools"; Flags: nowait postinstall skipifsilent
