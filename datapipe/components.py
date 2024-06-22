'''
Components
'''

from functions import parse_json, convert_type

import os
from collections import Counter
import re
import psycopg2

class Base:
    
    def __init__(self,
                 input_path : str = None,
                 schema : str = None,
                 table : str = None,
                 db_path_stg : str = None,
                 db_path_ods : str = None,
                 db_path_dmt : str = None,
                 sql_path_ods : str = None,
                 sql_path_dmt : str = None,
                 delimiter : str = None,
                 encoding_scheme : str = None,
                 max_file_read : int = None):
        
        self.input_path = input_path
        
        self.schema = schema
        
        self.table = table
        
        self.db_path_stg = db_path_stg
        
        self.db_path_ods = db_path_ods
        
        self.db_path_dmt = db_path_dmt
        
        self.sql_path_ods = sql_path_ods
        
        self.sql_path_dmt = sql_path_dmt
        
        self.delimiter = delimiter
        
        self.encoding_scheme = encoding_scheme if encoding_scheme else 'UTF-8'
        
        self.max_file_read = max_file_read if max_file_read else 1073741824



class FileReader(Base):
    
    def read_directory(self):
        
        files = os.listdir(self.input_path)
        
        self.structure = {file:file[file.rindex('.')+1:] \
                          for file in files if not os.path.isdir(file)}

            
            
class FileIterator(FileReader):

    def fetch_input(self):
        
        self.currentfile = None
        
        self.currentext = None
        
        for file,ext in self.structure.items():
            
            if 'TEMP_' in file:
            
                self.currentfile = file
                
                self.currentext = ext
                
            else:
                
                continue



class FileParser(FileIterator):
    
    def read_csv(self):
        
        fd = os.open(path=self.input_path+'/'+self.currentfile,
                     flags=os.O_RDONLY)

        content = os.read(fd,self.max_file_read)

        os.close(fd=fd)

        self.content = content.decode(self.encoding_scheme)
        
    def read_binary(self):
        
        #todo
        raise NotImplementedError
        
    def transform_to_matrix(self):
        
        nalnum = re.sub(r'[a-zA-Z0-9_\n]', '', self.content)

        cnt = Counter(nalnum)

        dlmtr = self.delimiter if self.delimiter else cnt.most_common(1)[0][0]

        self.data = [c.split(dlmtr) for c in self.content.split('\n')]
        
        for i, record in enumerate(self.data):
            
            for j, val in enumerate(record):

                self.data[i][j] = convert_type(val)


        
class DBConnector(FileParser):
    
    def connect(self,db_path : str):
        
        db_args = parse_json(db_path)
        
        self.connection = psycopg2.connect(**db_args)
    
    def disconnect(self):
        
        self.connection.close()
    


class SQLExecutor(DBConnector):
    
    def execute_sql(self,sql_path : str):
            
        fd = os.open(path=sql_path,
                         flags=os.O_RDONLY)

        content = os.read(fd,self.max_file_read)

        os.close(fd=fd)

        statement = content.decode(self.encoding_scheme)
        
        cursr = self.connection.cursor()
        
        cursr.execute(statement)
        
        self.connection.commit()
        
    def insert_records(self):
        
        for values in self.data:
            
            values = tuple(values)
            
            cursr = self.connection.cursor()
            
            cursr.execute(f'INSERT INTO {self.schema}.{self.table} VALUES{values};'\
                          .replace('None','NULL'))
        
            self.connection.commit()



class Pipeline(SQLExecutor):
    
    def __init__(self,kwargs):
        
        super(Pipeline,self).__init__(**kwargs)
        
        if self.db_path_stg:
        
            self.read_directory()
            
            self.fetch_input()
    
            if self.currentext in ['txt','csv']:
                
                self.read_csv()
            
            elif self.currentext in ['bin','eibcdic']:
                
                self.read_binary()
            
            self.transform_to_matrix()
        
            self.connect(self.db_path_stg)
          
            self.insert_records()
            
            self.disconnect()
            
        if self.db_path_ods:
        
            self.connect(self.db_path_ods)
            
            try:
            
                self.execute_sql(self.sql_path_ods)
                
            except:
                
                pass
            
            self.disconnect()
            
        if self.db_path_dmt:
        
            self.connect(self.db_path_dmt)
            
            try:
            
                self.execute_sql(self.sql_path_dmt)
            
            except:
                
                pass
            
            self.disconnect()
