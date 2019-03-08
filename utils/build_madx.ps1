#Requires -Version 3.0
$ErrorActionPreference = "Stop"

# On some machines required to make HTTPS work:
[Net.ServicePointManager]::Expect100Continue = $true;
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;

$MADXDIR = if ($env:MADXDIR) { $env:MADXDIR } else { "madx-bin" }
$MADXDIR = [System.IO.Path]::Combine($(pwd), $MADXDIR)

$MADX_VER = "5.04.02"
$MADX_ZIP = "$MADX_VER.zip"
$MADX_URL = "https://github.com/MethodicalAcceleratorDesign/MAD-X/archive/"
$MADX_DIR = "MAD-X-$MADX_VER"

$GC_VER = "8.0.2"
$GC_ZIP = "v$GC_VER.zip"
$GC_URL = "https://github.com/ivmai/bdwgc/archive/"
$GC_DIR = "$MADX_DIR\libs\gc\gc-$GC_VER"

function call()
{
    & $args[0] $args[1..$args.length]
    if (!$?) { throw "Exit code $LastExitCode from command `"$args`"." }
}

conda create -qf -n madx python=3.4 patch cmake
conda activate madx
conda install -q -c conda-forge mingwpy

$web = New-Object System.Net.WebClient
$web.DownloadFile("$MADX_URL/$MADX_ZIP", $MADX_ZIP)
$web.DownloadFile("$GC_URL/$GC_ZIP", $GC_ZIP)
Expand-Archive -LiteralPath $MADX_ZIP -DestinationPath .
Expand-Archive -LiteralPath $GC_ZIP -DestinationPath .
mv "bdwgc-$GC_VER" $GC_DIR

# Patch garbage collector to newer version, see #41:
call patch -d $MADX_DIR -p1 -i "$PSScriptRoot\patches\gc-8.0.2.diff"

# Build MAD-X as library:
mkdir "$MADX_DIR\build"
cd "$MADX_DIR\build"
call cmake .. -G 'MinGW Makefiles' `
    -DMADX_ONLINE=OFF `
    -DMADX_INSTALL_DOC=OFF `
    "-DCMAKE_INSTALL_PREFIX=$MADXDIR" `
    -DCMAKE_BUILD_TYPE=Release `
    -DMADX_STATIC=ON `
    -DBUILD_SHARED_LIBS=OFF

call mingw32-make install
