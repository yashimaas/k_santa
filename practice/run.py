from gurobipy import *
m = read("santa_sm.lp")
m.setParam("NodefileStart", 2.0)
setParam("MIPFocus", 0)
m.read("kernel_init.mst")
m.optimize()
m.write("s_r.lp")
m.write("s_r.mst")
m.write("s_r.sol")
