Developer setup:
- Install CUDA 8.0 for NVIDIA development

Maybe necessary if you lack ~/.nuget/packages and get an error about it being missing from Fody
- Right-click References > Manage NuGet packages
-- Uninstall Alea.Fody and Fody
-- Restart Visual Studio
-- Reinstall Alea.Fody (which installs the right version of Fody)
