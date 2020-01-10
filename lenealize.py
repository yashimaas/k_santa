from __future__ import print_function
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from numba import njit
from itertools import product

data_org = pd.read_csv('./data/family_data.csv', index_col='family_id')
data = data_org.iloc[0:1000,:]

N_DAYS = 100
N_FAMILIES = len(data)
MAX_OCCUPANCY = 60
MIN_OCCUPANCY = 20
TOP_K = 10

FAMILY_SIZE = data.n_people.values
DESIRED     = data.values[:, :-1] - 1

def get_penalty(n, choice):
    penalty = None
    if choice == 0:
        penalty = 0
    elif choice == 1:
        penalty = 50
    elif choice == 2:
        penalty = 50 + 9 * n
    elif choice == 3:
        penalty = 100 + 9 * n
    elif choice == 4:
        penalty = 200 + 9 * n
    elif choice == 5:
        penalty = 200 + 18 * n
    elif choice == 6:
        penalty = 300 + 18 * n
    elif choice == 7:
        penalty = 300 + 36 * n
    elif choice == 8:
        penalty = 400 + 36 * n
    elif choice == 9:
        penalty = 500 + 36 * n + 199 * n
    else:
        penalty = 500 + 36 * n + 398 * n
    return penalty

def GetPreferenceCostMatrix(data):
    cost_matrix = np.zeros((N_FAMILIES, N_DAYS), dtype=np.int64)
    for i in range(N_FAMILIES):
        desired = data.values[i, :-1]
        cost_matrix[i, :] = get_penalty(FAMILY_SIZE[i], 10)
        for j, day in enumerate(desired):
            cost_matrix[i, day-1] = get_penalty(FAMILY_SIZE[i], j)
    return cost_matrix

PCOSTM = GetPreferenceCostMatrix(data)


# @njit(fastmath=True)
def pcost(prediction):
    daily_occupancy = np.zeros((N_DAYS+1,), dtype=np.int64)
    penalty = 0
    for (i, p) in enumerate(prediction):
        n = FAMILY_SIZE[i]
        penalty += PCOSTM[i, p]
        daily_occupancy[p] += n
    return penalty, daily_occupancy

def GetAccountingCostMatrix():
    ac = np.zeros((MAX_OCCUPANCY+1, MAX_OCCUPANCY+1), dtype=np.float64)
    for n in range(MIN_OCCUPANCY,ac.shape[0]):
        for n_p1 in range(MIN_OCCUPANCY,ac.shape[1]):
            diff = abs(n - n_p1)
            ac[n, n_p1] = max(0, (n - MIN_OCCUPANCY) / 400 * n**(0.5 + diff / 50.0))
#             ac[n, n_p1] = max(0, (n - 2) / 8 * n**(0.5 + diff))
    return ac

ACOSTM = GetAccountingCostMatrix() 

@njit(fastmath=True)
def acost(daily_occupancy):
    accounting_cost = 0
    n_out_of_range = 0
    daily_occupancy[-1] = daily_occupancy[-2]
    for day in range(N_DAYS):
        n_p1 = daily_occupancy[day + 1]
        n    = daily_occupancy[day]
        n_out_of_range += (n > MAX_OCCUPANCY) or (n < MIN_OCCUPANCY)
        accounting_cost += ACOSTM[n, n_p1]
    return accounting_cost, n_out_of_range

@njit(fastmath=True)
def cost_function(prediction):
    penalty, daily_occupancy = pcost(prediction)
    accounting_cost, n_out_of_range = acost(daily_occupancy)
    return penalty + accounting_cost + n_out_of_range*100000000


def eval(prediction):
    pc, occ = pcost(prediction)
    ac, _ = acost(occ)
    print('Preferenced Cost : ', pc)
    print('Accounting Cost : {: .2f}'.format(ac))
    print('Total Cost : {: .2f}'.format(pc+ac))
    print('')
    print('Max Occupancy : {} , Min Ocupancy : {}'.format(occ.max(), occ.min()))
    


def IP():
    
    S = pywraplp.Solver('SolveAssignmentProblem', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
#     S = pywraplp.Solver('SolveAssignmentProblem', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)    
    GAP = MAX_OCCUPANCY-MIN_OCCUPANCY + 1
    
    candidates = [[] for _ in range(N_DAYS)] 
    
    x = {}
    for i in range(N_FAMILIES):
        for j in DESIRED[i, :]:
            candidates[j].append(i)
            x[i, j] = S.BoolVar('x[%i,%i]' % (i, j))
    
    N = {}
    for day in range(N_DAYS):
        for i in range(GAP):
            for j in range(GAP):
                N[day, i+MIN_OCCUPANCY, j+MIN_OCCUPANCY] = S.BoolVar('N[%i,%i,%i]' % (day, i+MIN_OCCUPANCY, j+MIN_OCCUPANCY))
            

    family_presence = [S.Sum([x[i, j] for j in DESIRED[i, :]])
                                                        for i in range(N_FAMILIES)]
    
    linear_constraint = [S.Sum(N[day, i+MIN_OCCUPANCY, j+MIN_OCCUPANCY] for i in range(GAP)
                                                                                                                               for j in range(GAP)) 
                                                                                                                               for day in range(N_DAYS) ]
    
    daily_occupancy_x = [S.Sum([x[i, j] * FAMILY_SIZE[i] for i in candidates[j]])
                                                                                      for j in range(N_DAYS)]
    
#     daily_occupancy_n = [S.Sum( [(i + MIN_OCCUPANCY)* N[day, i+MIN_OCCUPANCY, j+MIN_OCCUPANCY] for i in range(GAP)
#                                                                                                                                                                            for j in range(GAP)] )
#                                                                                                                                                                            for day in range(N_DAYS)]
    
 
        
    # Objective    
    partial_costs = [PCOSTM[i, j] * x[i,j] for i in range(N_FAMILIES) for j in DESIRED[i, :] ]
    preference_cost = S.Sum(partial_costs)
    
    penalties = []
    for day in range(N_DAYS):
        for i in range(GAP):
            for j in range(GAP):
                daily_occupancy = i + MIN_OCCUPANCY
                p_occupancy = j + MIN_OCCUPANCY
                penalties.append((daily_occupancy - 125)/400 * daily_occupancy**(1/2+abs(daily_occupancy - p_occupancy)/50) * N[day, daily_occupancy, p_occupancy])
#                 penalties.append((daily_occupancy - MIN_OCCUPANCY)/8 * daily_occupancy**(1/2+abs(daily_occupancy - p_occupancy)) * N[day, daily_occupancy, p_occupancy])

    accounting_penalty = S.Sum(penalties)
    
    total_cost = S.Sum([preference_cost,accounting_penalty])
#     total_cost = preference_cost + accounting_penalty

    S.Minimize(total_cost)
#     S.Minimize(accounting_penalty)
#     S.Minimize(preference_cost)


# 
    # Constraints
            #         差の条件を加えてみる
    for j in range(N_DAYS-1):
        S.Add(daily_occupancy_x[j]   - daily_occupancy_x[j+1] <= 23)
        S.Add(daily_occupancy_x[j+1] - daily_occupancy_x[j]   <= 23)
        
        
    for day in range(N_DAYS):
        S.Add(linear_constraint[day] == 1)
        S.Add(
#         daily_occupancy_x[day] - daily_occupancy_n[day] == 0
            S.Sum([N[day, i+MIN_OCCUPANCY, j+MIN_OCCUPANCY]*(i+MIN_OCCUPANCY) for i in range(GAP) for j in range(GAP)]) 
                == daily_occupancy_x[day]
        )       

    for day in range(N_DAYS-1):
        S.Add(
            S.Sum([N[day, i+MIN_OCCUPANCY, j+MIN_OCCUPANCY]*(j+MIN_OCCUPANCY) for i in range(GAP) for j in range(GAP)]) 
                == daily_occupancy_x[day+1]
        ) 
        
#         S.Add(daily_occupancy_n[day] == 0)
        
    for i in range(N_FAMILIES):
        S.Add(family_presence[i] == 1)
            
        

    print('ready !!')
    S.EnableOutput() 
    S.SetNumThreads(24)
#     S.SuppressOutput()
#     S.SetTimeLimit(1000*60*24**24)
    res = S.Solve()

    resdict = {0:'OPTIMAL', 1:'FEASIBLE', 2:'INFEASIBLE', 3:'UNBOUNDED', 
               4:'ABNORMAL', 5:'MODEL_INVALID', 6:'NOT_SOLVED'}

    print('IP solver result:', resdict[res])


    l = [(i, j, x[i, j].solution_value()) for i in range(N_FAMILIES)
                                                      for j in DESIRED[i, :] 
                                                      if x[i, j].solution_value()>0]
    

    df = pd.DataFrame(l, columns=['family_id', 'day', 'n'])
    return df


df_tmp = IP()

eval(df_tmp.day.values)

pc, occ = pcost(df_tmp.day.values)
print(occ)

import pickle
f = open('list.pkl', 'wb')
pickle.dump(df_tmp.day.values, f)
print(df_tmp.day.values)