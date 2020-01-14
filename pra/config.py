from gurobipy import *
m = read("s_r.mps")

m.NodefileStart = 2.0
m.MIPFocus = 0
m.Threads = 24


