$urls = @(
"https://aic-data.ledo.io.vn/Keyframes_L21.zip"
"https://aic-data.ledo.io.vn/Keyframes_L22.zip"
"https://aic-data.ledo.io.vn/Keyframes_L23.zip"
"https://aic-data.ledo.io.vn/Keyframes_L24.zip"
"https://aic-data.ledo.io.vn/Keyframes_L25.zip"
"https://aic-data.ledo.io.vn/Keyframes_L26_a.zip"
"https://aic-data.ledo.io.vn/Keyframes_L26_b.zip"
"https://aic-data.ledo.io.vn/Keyframes_L26_c.zip"
"https://aic-data.ledo.io.vn/Keyframes_L26_d.zip"
"https://aic-data.ledo.io.vn/Keyframes_L26_e.zip"
"https://aic-data.ledo.io.vn/Keyframes_L27.zip"
"https://aic-data.ledo.io.vn/Keyframes_L28.zip"
"https://aic-data.ledo.io.vn/Keyframes_L29.zip"
"https://aic-data.ledo.io.vn/Keyframes_L30.zip"
"https://aic-data.ledo.io.vn/clip-features-32-aic25-b1.zip"
)

foreach ($url in $urls) {
    Write-Host "Downloading $url ..."
    curl.exe -L -O $url
}
