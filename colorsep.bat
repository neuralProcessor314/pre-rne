@echo off
REM Split every RGB PNG in data\raw into three grayscale PNGs (R, G, B) in data\sep.
REM Example: data\raw\bRD_L0.1_3.png -> data\sep\bRD_L0.1_3_c0.png, _c1.png, _c2.png
REM In the loop, %%F is the file and %%~nF is its name without ".png".
REM %%d in the output name becomes 0, 1, 2 for the R, G, B channels.

if not exist data\sep mkdir data\sep

for %%F in (data\raw\*.png) do (
    magick "%%F" -alpha off -channel RGB -separate -define png:color-type=0 "data\sep\%%~nF_c%%d.png"
)
