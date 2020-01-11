from gurobipy import *
m = read("santa_lb.lp")
m.setParam("NodefileStart", 0.5)
setParam("MIPFocus", 3)
#m.read("kernel_init.mst")
m.optimize()
#m.write("s_r.lp")
#m.write("s_r.mst")
#m.write("s_r.sol")
