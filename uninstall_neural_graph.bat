@echo off
setlocal
set SI_LOCATION=C:\Softimage\SOFT3D_4.0
if exist "%SI_LOCATION%\3D\custom\bin\NeuralGraph.dll" del "%SI_LOCATION%\3D\custom\bin\NeuralGraph.dll"
if exist "%SI_LOCATION%\3D\custom\model\NeuralGraph.cus" del "%SI_LOCATION%\3D\custom\model\NeuralGraph.cus"
echo NeuralGraph removed. Restart SOFTIMAGE^|3D.
pause
endlocal
