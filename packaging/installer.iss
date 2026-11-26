; Inno Setup script: turns dist\FaceGenerator (built by PyInstaller) into a
; single setup .exe. Build with packaging\build.ps1, which runs:
;   ISCC.exe /DAppVersion=<version> packaging\installer.iss
; Output: dist\installer\FaceGeneratorSetup-<version>.exe

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
#define AppName "StyleGAN Face Generator"
#define AppExe "FaceGenerator.exe"

[Setup]
; AppId identifies the app for upgrades and uninstall: never change it.
AppId={{7C3E5A1F-4B2D-4E8A-9F61-2D8C0B5E7A94}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
DefaultDirName={autopf}\FaceGenerator
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; Installs for the current user without admin rights by default; the user
; can choose "install for all users" (needs admin) in the first dialog.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=..\LICENSE.txt
SetupIconFile=facegen.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
OutputDir=..\dist\installer
OutputBaseFilename=FaceGeneratorSetup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\FaceGenerator\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[Code]
// The app keeps the downloaded model (~325 MB) and its logs in
// %LOCALAPPDATA%\FaceGenerator (see config.py). Offer to remove them on
// uninstall. Generated images (in the user's FaceGenerator\output folder)
// are never touched.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\FaceGenerator');
    if DirExists(DataDir) and not UninstallSilent then
      if MsgBox('Also delete the downloaded model and log files?' + #13#10#13#10 +
                DataDir + #13#10#13#10 +
                'Your generated images are kept either way.',
                mbConfirmation, MB_YESNO) = IDYES then
        DelTree(DataDir, True, True, True);
  end;
end;
