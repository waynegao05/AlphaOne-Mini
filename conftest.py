"""项目根 conftest，确保 pytest 可以从 ``game``、``evaluate`` 包中导入。"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
