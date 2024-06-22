'''
Author: Ntigkaris E. Alexandros
'''

import sys
from dependencies import collect_requirements

collect_requirements()

from functions import parse_json
from components import Pipeline

try:
    
    setup_arg = sys.argv[1]
    
except:
    
    raise ValueError('Invalid argument encountered. Please provide the full \
                     path corresponding to the setup.json file!')

setup_path = parse_json(setup_arg)
job = Pipeline(setup_path)
