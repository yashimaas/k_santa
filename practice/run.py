from gurobipy import *
m = read("santa_lb2.lp")
#m = read("s_r.mps")
m.setParam("NodefileStart", 1.0)
m.setParam("MIPFocus", 3)
m.read("s_r.mst")
m.setParam("Threads", 24)
m.update()
m.optimize()
m.write("s_r.lp")
m.write("s_r.mps")
m.write("s_r.mst")
m.write("s_r.sol")

