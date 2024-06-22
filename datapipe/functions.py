'''
Functions
'''

import ast

def parse_json(path : str):
    
    with open(path) as file:
        
        items = file.read()
        
    items = ast.literal_eval(items)
    
    return items.__getitem__(0)


def convert_type(value : str):
    
    if len(value) > 0:
    
        if value.isalnum():
            
            return value
        
        else:
            
            return float(value)
        
    else: return None
