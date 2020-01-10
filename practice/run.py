from gurobipy import *
m = read("santa_sm.lp")
m.setParam("NodefileStart", 2.0)
setParam("MIPFocus", 1)

m.optimize()
m.write("s_s.lp")
m.write("s_s.mst")
