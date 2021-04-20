from __future__ import print_function
import re

class TeePrint:
    file = None

    def __init__(self,filename=None,mode='a'):
        assert filename and isinstance(filename,str), 'Must specify a valid filename.'
        self.file = open(filename,mode)

    def __del__(self):
        if self.file:
            self.file.close()

    def write(self,msg,end='\n'):
        print(msg,end=end)
        self.file.write(re.sub(r'\033\[\S+m','',msg)+end)
        self.file.flush()


