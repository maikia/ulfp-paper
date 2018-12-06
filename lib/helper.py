import numpy as np
import os
import shutil

def remove_folder(folder_name):
    """ removes all the files from the given folder;
    folder_name: name of the folder from which to remove files (it will not remove subfolders)
    remove_from_subs: if True all the subfolders will be cleared as well"""
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    

def remove_content(folder_name):
    """ removes all the files from the given folder;
    folder_name: name of the folder from which to remove files 
    (it will not remove subfolders)"""    

    for filename in os.listdir(folder_name):
        filepath = os.path.join(folder_name, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)

def create_folder(folder_name):
    """ checks if given folder exists, if not it creates it"""
    
    # checks if the given folder/file exists and if not calculates the data
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        

def file_exists(folder_name, filename):  
    """ checks if given file exists in given folder, and returns True or False"""

    filepath = os.path.join(folder_name, filename)
    exists = os.path.isfile(filepath)

    return exists      

def find_files(dir_name, ext='swc'):
    ''' returns all the files with the given extansion in the directory dir'''
    import glob, os
    old_dir = os.getcwd()
    os.chdir(dir_name)
    files = []
    for file in glob.glob("*."+ext):
        files.append(file)
    os.chdir(old_dir)
    return files

def find_directories(dirname):
    dirs_in = [f for f in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, f))]
    return dirs_in

def calc_ab_line_given_pts(x1, x2, y1, y2):
    '''
    calculates a and b of the equation y=ax+b given coordinates of two points
    :param x1: x of first point
    :param x2: x of second point
    :param y1: y of first point
    :param y2: y of second point
    :return a: a of line
    :return b: b of line
    :return radiens: radiens of the line
    '''
    a_line = (y1 - y2) / (x1 - x2)
    b_line = y2 - a_line * x2
    radians = np.arctan(-1 / a_line)
    return a_line, b_line, radians


