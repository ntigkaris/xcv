'''
Dependencies
'''

import sys
import subprocess

required = ['psycopg2']

def collect_requirements():
    
    for m in required:
        
        if m in sys.modules:
            
            continue

        else:
            
            _ = subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install', m ])
