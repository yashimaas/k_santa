from gurobipy import *
m = read("santa.lp")
m.read("kernel_init.mst")
m.setParam("NodefileStart", 2.0)
setParam("MIPFocus", 0)

m.optimize()
m.write("santa_r.lp")
m.write("santa_r.mst")
m.write("santa_r.sol")
