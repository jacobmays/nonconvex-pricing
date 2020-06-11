#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 20:51:00 2017

@author: jacobmays
"""

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


alpha = 0.5
beta = 0.5
b = 10000.

c_fix_1 = 80.0
c_fix_2 = 50.0
c_run = 50.0

epsilon = c_run*(alpha + (1-alpha)*beta)/b
                
def objective(x):
    return -1.0*(-c_fix_1*x[0] - c_fix_2*x[1] \
            + (b)*(x[0]**2/2.0 + x[0]*(max(x[0]+epsilon*x[1],beta*x[1]) - x[0]) + (x[0]+x[1] - max(x[0]+epsilon*x[1],beta*x[1]))*(max(x[0]+epsilon*x[1],beta*x[1]) + x[0] + x[1])/2.0+(1-x[0]-x[1])*(x[0]+x[1])) \
            - c_run*alpha*x[1]*(1 - max(x[0]+epsilon*x[1],beta*x[1])) \
            - c_run*(1.0-alpha)*( (x[0]+beta*x[1] - max(x[0]+epsilon*x[1],beta*x[1]))*beta*x[1] \
                     + (x[1] - beta*x[1])*(beta*x[1] + x[1])/2.0 \
                       + (1.0 - x[0] - x[1])*x[1] ) )
            
bnd = (0.0,1.0)
bnds = (bnd,bnd)

x0 = np.zeros(2)
x0[0] = .7
x0[1] = .299

solution = minimize(objective,x0,method='SLSQP',bounds=bnds)

x_star = solution.x

print(x_star)
print(-objective(x_star))

x_2_num = -c_fix_1 + c_fix_2 + c_run*c_fix_1/b
x_2_denom = -2*c_run - b*epsilon**2 + c_run**2/b + 2*c_run*alpha*epsilon - c_run*(1-alpha)*(beta**2 - 2*beta*epsilon-1)

x_2 = x_2_num/x_2_denom
print(x_2)

alphas = np.arange(0,1.02,0.02)
x_2_shares = np.zeros(len(alphas))

for i in range(len(alphas)):
    epsilon = c_run*(alphas[i] + (1-alphas[i])*beta)/b
    x_2_num = -c_fix_1 + c_fix_2 + c_run*c_fix_1/b
    x_2_denom = -2*c_run - b*epsilon**2 + c_run**2/b + 2*c_run*alphas[i]*epsilon - c_run*(1-alphas[i])*(beta**2 - 2*beta*epsilon-1)
    x_2_shares[i] = x_2_num/x_2_denom

#fig1, ax1 = plt.subplots()

#ax1.plot(alphas, x_2_shares, 'k',label='Optimal size of generator 2')

#ax1.set_xlabel('$ \\alpha $')
#ax1.set_ylabel('Optimal size of generator 2')

epsilon = c_run*(alpha + (1-alpha)*beta)/b
                
beta_min = c_run*alpha/(b-c_run*(1-alpha))

betas = np.arange(0.02,1.02,0.02)
x_2_shares_beta = np.zeros(len(betas))

for i in range(len(betas)):
    epsilon = c_run*(alpha + (1-alpha)*betas[i])/b
    x_2_num = -c_fix_1 + c_fix_2 + c_run*c_fix_1/b
    x_2_denom = -2*c_run - b*epsilon**2 + c_run**2/b + 2*c_run*alpha*epsilon - c_run*(1-alpha)*(betas[i]**2 - 2*betas[i]*epsilon-1)
    x_2_shares_beta[i] = x_2_num/x_2_denom

#fig2, ax2 = plt.subplots()
#ax2.plot(betas, x_2_shares_beta, 'k',label='Optimal size of generator 2')

#ax2.set_xlabel('$ \\beta $')
#ax2.set_ylabel('Optimal size of generator 2')

epsilon = c_run*(alpha + (1-alpha)*beta)/b

# Profit under different pricing strategies
prob = np.zeros(5)
price = np.zeros(5)
dispatch = np.zeros((5,2))
make_whole = np.zeros(5)

price[1] = b
price[4] = b
 
method = 'LMP'
loc_method = 'NONE'
     
if method == 'LMP':
    price[2] = 0.0
    price[3] = c_run*(1-alpha)
    make_whole[2] = beta*(c_run*(1-alpha)-price[2]) + c_run*alpha
    make_whole[3] = (1+beta)*(c_run*(1-alpha)-price[3])/2 + c_run*alpha
elif method == 'CHP':
    price[1] = c_run
    price[2] = c_run
    price[3] = c_run
    make_whole[2] = beta*(c_run*(1-alpha)-price[2]) + c_run*alpha
    make_whole[3] = (1+beta)*(c_run*(1-alpha)-price[3])/2 + c_run*alpha
elif method == 'RLMP':
    price[2] = c_run*(1-alpha)
    price[3] = c_run*(1-alpha)
    make_whole[2] = beta*(c_run*(1-alpha)-price[2]) + c_run*alpha
    make_whole[3] = (1+beta)*(c_run*(1-alpha)-price[3])/2 + c_run*alpha
elif method == 'ELMP':
    price[2] = c_run
    price[3] = c_run
    make_whole[2] = beta*(c_run*(1-alpha)-price[2]) + c_run*alpha
    make_whole[3] = (1+beta)*(c_run*(1-alpha)-price[3])/2 + c_run*alpha
else:
    price[2] = c_run + c_run/beta
    price[3] = c_run*(1-alpha) + c_run*alpha*np.log(1/beta)/(1-beta)
        

def profit_no_makewhole(x,prices):
    prob[0] = x[0]
    prob[1] = max(x[0] + epsilon*x[1],beta*x[1]) - x[0]
    prob[2] = x[0] + beta*x[1] - max(x[0] + epsilon*x[1],beta*x[1]) 
    dispatch[2,0] = max(1.0/2.0,1.0 + (epsilon-beta)*x[1]/(2*x[0]))
    prob[3] = x[1]-beta*x[1]
    prob[4] = 1 - x[0] - x[1]
    return (prob[1]*price[1] + prob[2]*price[2]*dispatch[2,0] + prob[3]*price[3] + prob[4]*price[4] - c_fix_1, \
            - prob[2]*make_whole[2] - prob[3]*make_whole[3] + prob[4]*(price[4]-c_run) - c_fix_2)

def profit_no_makewhole_neg(x,prices):
    prob[0] = 0.0
    prob[1] = beta*x[1]
    prob[2] = 0.0
    dispatch[2,0] = 0.0
    prob[3] = x[1]-beta*x[1]
    prob[4] = max(0.0, 1 - x[1])
    return (0.0, \
            - prob[2]*make_whole[2] - prob[3]*make_whole[3] + prob[4]*(price[4]-c_run) - c_fix_2)


def profit_makewhole(x,prices):
    prob[0] = x[0]
    prob[1] = max(x[0] + epsilon*x[1],beta*x[1]) - x[0]
    prob[2] = x[0] + beta*x[1] - max(x[0] + epsilon*x[1],beta*x[1]) 
    dispatch[2,0] = max(1.0/2.0,1.0 + (epsilon-beta)*x[1]/(2*x[0]))
    prob[3] = x[1]-beta*x[1]
    prob[4] = 1 - x[0] - x[1]
    return (prob[1]*price[1] + prob[2]*price[2]*dispatch[2,0] + prob[3]*price[3] + prob[4]*price[4] - c_fix_1, \
            prob[4]*(price[4]-c_run) - c_fix_2)

def profit_makewhole_neg(x,prices):
    prob[0] = 0.0
    prob[1] = beta*x[1]
    prob[2] = 0.0
    dispatch[2,0] = 0.0
    prob[3] = x[1]-beta*x[1]
    prob[4] = max(0.0, 1 - x[1])
    return (0.0, \
            prob[4]*(price[4]-c_run) - c_fix_2)

def profit_loc(x,prices):
    prob[0] = x[0]
    prob[1] = max(x[0] + epsilon*x[1],beta*x[1]) - x[0]
    prob[2] = x[0] + beta*x[1] - max(x[0] + epsilon*x[1],beta*x[1]) 
    dispatch[2,0] = max(1.0/2.0,1.0 + (epsilon-beta)*x[1]/(2*x[0]))
    prob[3] = x[1]-beta*x[1]
    prob[4] = 1 - x[0] - x[1]
    return (prob[1]*price[1] + prob[2]*price[2] + prob[3]*price[3] + prob[4]*price[4] - c_fix_1, \
            prob[1]*(price[1]-c_run) + prob[4]*(price[4]-c_run) - c_fix_2)
       
def profit_loc_neg(x,prices):
    prob[0] = 0.0
    prob[1] = beta*x[1]
    prob[2] = 0.0
    dispatch[2,0] = 0.0
    prob[3] = x[1]-beta*x[1]
    prob[4] = max(0.0, 1 - x[1])
    return (0.0,prob[1]*(price[1]-c_run) + prob[4]*(price[4]-c_run) - c_fix_2)


if loc_method == 'MW':  
    x_equil = fsolve(profit_makewhole,[.5, .49],price,maxfev=10000) 
    profits = profit_makewhole(x_equil,price)
    if profits[0] < -.0001:
        x_equil[0] = 0.0
        x_equil = fsolve(profit_makewhole_neg,[0.0, .99],price,maxfev=10000)
    if x_equil[0] < 0.0001:
        x_equil[0] = 0.0
        x_equil = fsolve(profit_makewhole_neg,[0.0, .99],price,maxfev=10000)
    profits_at_opt = profit_makewhole(x_star,price)
elif loc_method == 'LOC':  
    x_equil = fsolve(profit_loc,[.5, .49],price,maxfev=10000)     
    profits = profit_loc(x_equil,price)    
    if profits[0] < -.0001:
        x_equil[0] = 0.0
        x_equil = fsolve(profit_loc_neg,[0.0, .99],price,maxfev=10000)    
    if x_equil[0] < 0.0001:
        x_equil[0] = 0.0
        x_equil = fsolve(profit_loc_neg,[0.0, .99],price,maxfev=10000)
    profits_at_opt = profit_loc(x_star,price)
else:  
    x_equil = fsolve(profit_no_makewhole,[.5, .49],price,maxfev=10000)     
    profits = profit_no_makewhole(x_equil,price)    
    if profits[0] < -.0001:
        x_equil[0] = 0.0
        x_equil = fsolve(profit_no_makewhole_neg,[0.0, .99],price,maxfev=10000)    
    if x_equil[0] < 0.0001:
        x_equil[0] = 0.0
        x_equil = fsolve(profit_no_makewhole_neg,[0.0, .99],price,maxfev=10000)
    profits_at_opt = profit_no_makewhole(x_star,price)

print(x_equil)
prob[0] = x_equil[0]
prob[1] = max(x_equil[0] + epsilon*x_equil[1],beta*x_equil[1]) - x_equil[0]
prob[2] = x_equil[0] + beta*x_equil[1] - max(x_equil[0] + epsilon*x_equil[1],beta*x_equil[1]) 
dispatch[2,0] = max(1.0/2.0,1.0 + (epsilon-beta)*x_equil[1]/(2*x_equil[0]))
prob[3] = x_equil[1]-beta*x_equil[1]
prob[4] = 1 - x_equil[0] - x_equil[1]

print(prob)

inv_cost = c_fix_1*x_equil[0] + c_fix_2*x_equil[1]
cleared_value = (b)*(x_equil[0]**2/2.0 + x_equil[0]*(max(x_equil[0]+epsilon*x_equil[1],beta*x_equil[1]) - x_equil[0]) + (x_equil[0]+x_equil[1] - max(x_equil[0]+epsilon*x_equil[1],beta*x_equil[1]))*(max(x_equil[0]+epsilon*x_equil[1],beta*x_equil[1]) + x_equil[0] + x_equil[1])/2.0+(1-x_equil[0]-x_equil[1])*(x_equil[0]+x_equil[1]))
production_cost = c_run*alpha*x_equil[1]*(1 - max(x_equil[0]+epsilon*x_equil[1],beta*x_equil[1])) \
            + c_run*(1.0-alpha)*( (x_equil[0]+beta*x_equil[1] - max(x_equil[0]+epsilon*x_equil[1],beta*x_equil[1]))*beta*x_equil[1] \
                     + (x_equil[1] - beta*x_equil[1])*(beta*x_equil[1] + x_equil[1])/2.0 \
                       + (1.0 - x_equil[0] - x_equil[1])*x_equil[1] )
    
objective_value = cleared_value - inv_cost - production_cost


avg_price = (inv_cost+production_cost)/(cleared_value/b)

if method == 'AIC':
    avg_sales_2 = prob[4]*price[4] + (c_run*alpha*x_equil[1]*(x_equil[0] + x_equil[1] - max(x_equil[0]+epsilon*x_equil[1],beta*x_equil[1])) \
            + c_run*(1.0-alpha)*( (x_equil[0]+beta*x_equil[1] - max(x_equil[0]+epsilon*x_equil[1],beta*x_equil[1]))*beta*x_equil[1] \
                     + (x_equil[1] - beta*x_equil[1])*(beta*x_equil[1] + x_equil[1])/2.0 ))/x_equil[1]
else:
    avg_sales_2 = prob[4]*price[4] + prob[2]*beta*price[2] + prob[3]*(1+beta)*price[3]/2
                      
avg_cost_2 = production_cost/x_equil[1]
avg_makewhole = np.dot(prob,make_whole)
avg_loc_1 = (1-dispatch[2,0])*prob[2]*price[2]
avg_loc_2 = prob[1]*(price[1]-c_run)

print(objective_value)
print(x_equil[0])
print(x_equil[1])
print(avg_price)
print(avg_loc_1)
print(avg_sales_2)
print(avg_makewhole)
print(avg_loc_2)
print(avg_cost_2)
print(profits_at_opt[0])
print(profits_at_opt[1])
print(prob[1]+prob[4])
