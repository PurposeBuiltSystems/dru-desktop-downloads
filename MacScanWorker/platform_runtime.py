"""Process-lifetime locks and active-work sleep prevention on supported desktops."""
import os
import sys


def lock_file(stream):
    if os.name == 'nt':
        import msvcrt
        stream.seek(0)
        if not stream.read(1):
            stream.write('0')
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


def active_command(command):
    return ['/usr/bin/caffeinate', '-i'] + command if sys.platform == 'darwin' else command
