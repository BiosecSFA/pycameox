"""
This module provides constants and other package-wide stuff.

"""

from pathlib import Path
from typing import Dict, Counter, NewType, Union

import pandas as pd

# Licence notice
LICENSE = ('''

    Copyright (C) 2022–2023, Jose Manuel Martí (LLNL)
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.
    
    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    
''')

# ### Type annotations
Filename = NewType('Filename', Path)
Id = NewType('Id', int)
Seq = NewType('Seq', str)
RunsSet = NewType('RunsSet', Dict[str, pd.DataFrame])
SampleSet = NewType('Sample', Dict[str, pd.DataFrame])

# ### Constants
HHFILTER:Path = Path('/usr/local/bin/hhfilter')
