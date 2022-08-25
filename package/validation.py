from genericpath import isfile
import os
import mimetypes

def validation_generate(num, path1, path2, output_txt=None):
    if os.path.isdir(path1) == False or os.path.isdir(path2) == False:
        raise TypeError('The input path should be vaild document')
    if type(num) != int:
        raise TypeError("The number of image should be number")
    if num <= 0:
        raise ValueError("The number of image should be positive number")
    if output_txt != None:
        if os.path.isfile(output_txt) == False:
            raise TypeError('The type of document should be text')

def validation_loader(path):
    if os.path.isfile(path) == False:
        raise TypeError('The input path should be vaild document')
    
    
        

