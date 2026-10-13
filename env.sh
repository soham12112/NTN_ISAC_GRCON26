# Source this before running anything Python in this project.
#
# Portable across the two hosts the OTA setup needs. The interpreter is
# whichever one GNU Radio was built against, which is not the same on both:
#
#   Linux box  GNU Radio 3.10.1.1  /usr/bin/python3 (3.10)
#              /usr/local/bin/python3 (3.12) shadows PATH and has no GNU Radio,
#              and PYTHONPATH carries ROS Humble, whose numpy conflicts.
#   macOS      GNU Radio 3.10.12.0 via Homebrew, python3.14, plus a venv at
#              .venv (created with --system-site-packages) for scipy, pyqtgraph
#              and pyzmq, which Homebrew does not supply.
#
# The OOT install tree also lands in a different place per platform, so both
# are globbed rather than hardcoded.

NTN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export NTN_ROOT
export NTN_DATA="$NTN_ROOT/data"
export NTN_OOT="$NTN_ROOT/gr-ntnisac/install"

case "$(uname -s)" in
Darwin)
    if [ -x "$NTN_ROOT/.venv/bin/python" ]; then
        export NTN_PY="$NTN_ROOT/.venv/bin/python"
    else
        export NTN_PY="$(command -v python3.14 || command -v python3)"
    fi
    _ntn_libvar=DYLD_LIBRARY_PATH
    ;;
*)
    export NTN_PY=/usr/bin/python3
    _ntn_libvar=LD_LIBRARY_PATH
    ;;
esac

# Everything the OOT installs, wherever CMake decided to put it: site-packages
# on macOS, dist-packages under local/ on Debian. Discovered with find rather
# than a glob because zsh treats an unmatched glob as an error, and this file
# is sourced from zsh on the Mac and bash on the Linux box.
_ntn_pypath="$NTN_ROOT/py:$NTN_ROOT/gr"
for d in $(find "$NTN_OOT" -maxdepth 4 -type d \
                \( -name site-packages -o -name dist-packages \) 2>/dev/null); do
    _ntn_pypath="$_ntn_pypath:$d"
done
export PYTHONPATH="$_ntn_pypath"

_ntn_libpath=""
for d in $(find "$NTN_OOT" -maxdepth 2 -type d -name 'lib' -o \
                -maxdepth 2 -type d -name '*-linux-gnu' 2>/dev/null); do
    _ntn_libpath="${_ntn_libpath:+$_ntn_libpath:}$d"
done
if [ -n "$_ntn_libpath" ]; then
    eval "export $_ntn_libvar=\"$_ntn_libpath\${$_ntn_libvar:+:\$$_ntn_libvar}\""
fi

export GRC_BLOCKS_PATH="$NTN_OOT/share/gnuradio/grc/blocks"
unset _ntn_pypath _ntn_libpath _ntn_libvar
alias ntnpy='$NTN_PY'
