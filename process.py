import sys
import ctypes
import subprocess
import os

def run(cmd, hide_stderr = True):
    libc_search_dirs = ["/lib", "/lib/x86_64-linux-gnu", "/lib/powerpc64le-linux-gnu"]

    if sys.platform=="linux" :
        found = None
        for d in libc_search_dirs:
            file = os.path.join(d, "libc.so.6")
            if os.path.isfile(file):
                found = file
                break

        if not found:
            print("WARNING: Cannot find libc.so.6. Cannot kill process when parent dies.")
            killer = None
        else:
            libc = ctypes.CDLL(found)
            PR_SET_PDEATHSIG = 1
            KILL = 9
            killer = lambda: libc.prctl(PR_SET_PDEATHSIG, KILL)
    else:
        print("WARNING: OS not linux. Cannot kill process when parent dies.")
        killer = None


    if hide_stderr:
        stderr = open(os.devnull,'w')
    else:
        stderr = None

    return subprocess.Popen(cmd.split(" "), stderr=stderr, preexec_fn=killer)
