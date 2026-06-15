@echo off
echo Creating Desktop Shortcut...

:: Create a VBS script for a desktop shortcut
echo Set oWS = WScript.CreateObject("WScript.Shell") > CreateShortcut.vbs
echo sLinkFile = "%USERPROFILE%\Desktop\ESCO Knowledge Graph.lnk" >> CreateShortcut.vbs
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> CreateShortcut.vbs
echo oLink.TargetPath = "python.exe" >> CreateShortcut.vbs
echo oLink.Arguments = "-m streamlit run esco_kg_streamlit.py" >> CreateShortcut.vbs
echo oLink.WorkingDirectory = "%~dp0" >> CreateShortcut.vbs
echo oLink.Description = "ESCO Knowledge Graph Application" >> CreateShortcut.vbs
echo oLink.IconLocation = "python.exe,0" >> CreateShortcut.vbs
echo oLink.Save >> CreateShortcut.vbs

:: Run VBScript
cscript //nologo CreateShortcut.vbs

:: Delete temporary VBS script
del CreateShortcut.vbs

echo Desktop shortcut created successfully!
pause 