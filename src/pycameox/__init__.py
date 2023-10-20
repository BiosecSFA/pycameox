#
#     Copyright (C) 2022, 2023, Jose Manuel Martí Martínez (LLNL)
#
"""
Python library for CAMEOX: CAMEOs eXtended
"""

__all__ = ['config', 'downstream',
           '__author__', '__date__', '__version__']
__author__ = 'Jose Manuel Martí'
__copyright__ = 'Copyright (C) 2022, 2023, Jose Manuel Martí (LLNL)'
__license__ = 'AGPLv3'
__maintainer__ = 'Jose Manuel Martí'
__status__ = 'Alpha'
__date__ = 'Oct 2023'
__version__ = '0.2.1'

import sys

# python
MAJOR, MINOR, *_ = sys.version_info
PYTHON_REL = (MAJOR == 3 and MINOR >= 8)
if not PYTHON_REL:
    raise ImportError('pyCAMEOX requires Python 3.8 or later')
