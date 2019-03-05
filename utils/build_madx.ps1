#Requires -Version 3.0
$MADXDIR = if ($env:MADXDIR) { $env:MADXDIR } else { "madx-bin" }
$MADXDIR = [System.IO.Path]::Combine($(pwd), $MADXDIR)

# Acquire MAD-X:
$MADX_VER = "5.04.02"
$MADX_ZIP = "$MADX_VER.zip"
$MADX_URL = "https://github.com/MethodicalAcceleratorDesign/MAD-X/archive/"
$MADX_DIR = "MAD-X-$MADX_VER"
& call python -m wget "$MADX_URL/$MADX_ZIP" -o $MADX_ZIP
& call 7za x $MADX_ZIP

:: Patch garbage collector to newer version, see #41:
$GC_VER = "8.0.2"
$GC_ZIP = "gc-$GC_VER.tar.gz"
$GC_URL = "https://github.com/ivmai/bdwgc/releases/download/"
& python -m wget "$GC_URL/v$GC_VER/$GC_ZIP" -o $GC_ZIP
& 7za x $GC_ZIP
& 7za x -o$MADX_DIR\libs\gc gc-$GC_VER.tar
& patch -d $MADX_DIR -p1 -i "$PSScriptRoot\patches\gc-8.0.2.diff"

# Build MAD-X as library:
mkdir "$MADX_DIR\build"
cd "$MADX_DIR\build"
& cmake .. -G 'MinGW Makefiles' `
    -DMADX_ONLINE=OFF `
    -DMADX_INSTALL_DOC=OFF `
    -DCMAKE_INSTALL_PREFIX=$MADXDIR `
    -DCMAKE_BUILD_TYPE=Release `
    -DMADX_STATIC=ON `
    -DBUILD_SHARED_LIBS=OFF

& mingw32-make install
