from gurobipy import *
m = read("santa_lb.lp")
m.setParam("NodefileStart", 0.5)
setParam("MIPFocus", 3)
#m.read("s_r.mst")
#m.optimize()

# set lower_bound
expr = LinExpr()
for v in m.getVars():
    expr.addTerms(v.Obj,v)
    #print((v.Obj,v))
m.addConstr(expr >= 7776.0,"lower_bound")

# end set
m.optimize()
m.write("lb_r.lp")
m.write("lb_r.mst")
m.write("lb_r.sol")
