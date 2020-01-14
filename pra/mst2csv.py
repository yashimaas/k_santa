import numpy as np
import pandas as pd
from numba import njit
from itertools import product
from ortools.linear_solver import pywraplp
from multiprocessing import Process
import re

import sys

def main(path,path_to):
    template = pd.read_csv('../data/sample_submission.csv')
    f = open(path, 'r')
    solution = f.read()
    # solution = solution.split('\n')[1:]
    f.close()
    pattern = 'x\[(\d+),(\d+)\] 1'
    rep = re.compile(pattern)
    ans = re.findall(rep,solution)
    submit = template.copy()
    for x in ans:
        submit.iat[int(x[0]),1] = int(x[1])+1
    submit.to_csv(path_to,index=False)
    

if __name__ == '__main__':
    args = sys.argv
    if len(args) > 2:
        main(args[1],args[2])
    else:
        print('Arguments are too short')
    
    
