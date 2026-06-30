import os
import sys
import platform


# ===== Function: print_to_terminal =====
def print_to_terminal(msg):
    if platform.system() == "Windows":
        sys.__stdout__.write(msg + '\n')
    else:
        with open('/dev/tty', 'w') as f:
            f.write(msg + '\n')