# WARNING : This method is not ideal.
from inspect import getsourcefile
import os
from pathlib import Path

path = Path(os.path.abspath(getsourcefile(lambda:0)))
fn = os.path.join(path.parent.parent.parent, 'reconstructor/utils/logger.py')

with open(fn, 'r') as f:
    code = f.read()

exec(code)
