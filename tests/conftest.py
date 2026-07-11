"""pytest yapılandırması: repo kökünü import yoluna ekler.

Böylece `pytest tests/` hangi dizinden koşulursa koşulsun
`import spine` ve `import materials` çalışır.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
