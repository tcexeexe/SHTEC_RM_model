#******************************************************************************************************** 
#  @author	     tcexeexe
#  @date         04,2024
#
#  @par     Copyright (c) 2024, SHTEC
# 
# *******************************************************************************************************/
"""Safe-RLHF: Safe Reinforcement Learning with Human Feedback."""

__version__ = '0.0.1dev0'
__license__ = 'Apache License, Version 2.0'
__author__ = 'PKU-Alignment Team'
__release__ = False

if not __release__:
    import os
    import subprocess

    try:
        prefix, sep, suffix = (
            subprocess.check_output(
                ['git', 'describe', '--abbrev=7'],  # noqa: S603,S607
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
            .lstrip('v')
            .replace('-', '.dev', 1)
            .replace('-', '+', 1)
            .partition('.dev')
        )
        if sep:
            version_prefix, dot, version_tail = prefix.rpartition('.')
            prefix = f'{version_prefix}{dot}{int(version_tail) + 1}'
            __version__ = sep.join((prefix, suffix))
            del version_prefix, dot, version_tail
        else:
            __version__ = prefix
        del prefix, sep, suffix
    except (OSError, subprocess.CalledProcessError):
        pass

    del os, subprocess
