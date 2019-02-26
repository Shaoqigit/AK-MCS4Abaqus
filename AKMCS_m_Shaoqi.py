# -*- coding: utf-8 -*-
"""
Created on Wed May 02 09:39:53 2018
20/02/2019 modified by shaoqi WU about eveluation points
21/02/2019 modified by shaoqi WU about sigma and calculation of U
25/02/2019 modified by Shaoqi WU about sigma calculated directly in library Pykriging: predict_var
there are different libraries of meta model kriging in internet, we dont know which one? 
"""

import numpy as np
from numpy import cosh, sinh

import pyKriging  
from pyKriging.krige import kriging

import random

import matplotlib.pyplot as plt

from scipy.optimize import minimize, root

import time

start = time.clock()
# *** Joint properties ***#
# Constants
b = 30.             # Joint width [mm]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
l = 60.             # Joint lengthlongueur du joint [mm]
Ep = 62000.         # Young modulus of joined parts [MPa]
ep1 = 4.            # Part 1 thickness [mm]
ep2 = 4             # Part 2 thickness [mm]
F = 6000.          # Applied force [N]
Tau_M = F/(b*l)     # Mean shear stress [MPa]
Tau_R = 20.        # Shear resistance


def volkersen(INPUT):
    X = np.arange(-l/2., l/2.+l/50., l/50.)
    
    ea1 = INPUT[0]
    Ga1 = INPUT[1]

    ea = convertCtrReal(ea1,aE,bE)
    Ga = convertCtrReal(Ga1,aG,bG)
    
    w = ((Ga/ea)*(1./(Ep*ep1)+1./(Ep*ep2)))**0.5
    Tau = Tau_M *(w*l/2.)*((cosh(w*X)/sinh(w*l/2.))+((Ep*ep1 - Ep*ep2)/(Ep*ep1 + Ep*ep2))*(sinh(w*X)/cosh(w*l/2.)))
    Tau_Max = np.max(Tau)
    return Tau_Max-Tau_R

    
def plot(X, Y):
    minX = int(min(X)/10)*10-10
    maxX = int(max(X)/10)*10+10
    minY = int(min(Y)/10)*10-10
    maxY = int(max(Y)/10)*10+20
    plt.figure(figsize=(10,10))
    plt.plot(X, Y, 'r-o', label='Volkersen',linewidth=2)
    plt.xticks(range(minX,maxX+10,10))
    plt.xlim(minX, maxX)
    plt.xlabel("Distance [mm]",fontsize=22)
    plt.yticks(range(minY,maxY+10,10))
    plt.ylim(minY, maxY)
    plt.ylabel("Shear stress [MPa]",fontsize=20)
    plt.tick_params(axis='both',labelsize=22)
#    plt.legend(loc='upper left',fontsize=18)
#    plt.grid()
        

def samplingMC(nMC):
    X = []
    for i in range(nMC):
        random.seed(a=None)
        x = random.gauss(0,1)
        X.append(x)
    return X

def plotSamples(S, area, color):
    try:
        X1, X2 = [], []
        for s in S:
            X1.append(s[0])
            X2.append(s[1])
        
        plt.scatter(X1, X2, s = area, c = color)
        plt.xticks(range(-3,4,1))
        plt.xlim(-3, 3)
        plt.xlabel("Thickness",fontsize=22)
        plt.yticks(range(-3,4,1))
        plt.ylim(-3, 3)
        plt.ylabel("Shear modulus",fontsize=20)
        plt.tick_params(axis='both',labelsize=22)
    except:
        pass

def plotFunction(X, Y, Line, Label):
    plt.plot(X,Y,Line,label=Label,linewidth=2,markersize=8)    
    
def getInitialPop(S, nMC, N1):
    Indices = []
    S1, S2 = [], []
    for i in range(1000*nMC):
        random.seed(a=None)
        index = random.randint(0,nMC)
        if index not in Indices and len(Indices)<N1:
            Indices.append(index)
            S1.append(S[index])
        if len(Indices) == N1:
            break
       
    S2 = np.delete(S,Indices,axis=0)
    return np.array(S1), np.array(S2), np.array(Indices)
    

def convertCtrReal(X, a, b):
    return a*X+b

def funcFit(X, r1, r2):
    return r1*X+r2

def funcSolPoly(x):
    maxDeg = len(POLY)
    value = 0
    for i in range(maxDeg):
        deg = float(maxDeg-i-1)
        coef = POLY[i]
        value += coef*x**deg
    return value
    

# ------------------------------- Main scipt ---------------------------------# 
# define sampling size
nMC = 5000

# define initial number of individuals to be used to build kriging model 
N1 = 5

# sampling nMC value of e0 and G0
E0 = np.array(samplingMC(nMC))
G0 = np.array(samplingMC(nMC))
#plt.scatter(E0, G0)

# convert to real variation domain
aE, bE = 0.2, 1.
aG, bG = 1000., 4000.  # ?
E = convertCtrReal(E0,aE,bE)
G = convertCtrReal(G0,aG,bG)

# reshape inputs
S0 = np.reshape([E0, G0],(nMC,2))



# get initial population (N1 individual)
S1, S2, Indices1 = getInitialPop(S0, nMC, N1)
#sigma = np.std(S2)
#print(sigma)

# *********************** BUILDING INITIAL KRIGING MODEL *********************#
Y1 = []
for i in range(N1):
    s1 = S1[i]
    y1 = volkersen(s1)
    Y1.append(y1)
    
#print(S1)
#print(Y1)
k = kriging(S1, Y1, testfunction=volkersen, name='simple')
k.train()
#print(k.X)



# ********************************** PLOT ************************************#
print("\n****  Initialization  ****")

maxIter = 15
NEW_S1 = S1
NEW_S2 = S2

#umin = 0

# calculate failure probability
S_POS = []
S_NEG = []
S_PRE = []
for s in S2:
    ev_s = k.predict(s)
    S_PRE.append(ev_s)

    if ev_s<=0:
        S_NEG.append(s)
    else:
        S_POS.append(s)

pf = float(len(S_NEG))/float(len(S0))
print("failure probability= "+str(pf))
#sigma_ini = np.std(S_PRE)
#print("sigma G(s)"+str(np.std(S_PRE)))   
# plot figures
plt.figure(figsize=(10,10))
plotSamples(S_POS, 10, 'r')
plotSamples(S_NEG, 10, 'g')
plotSamples(S1, 80, 'b')

#X_plot1 = np.arange(-3., 3.1, 0.1)
#X_plot2 = [] 
#nPlot = len(X_plot1)
#
#for i in range(nPlot):
#    Y_SOL = []
#    for j in range(nPlot):
#        s_predict = [X_plot1[i],X_plot1[j]]
#        y_sol = k.predict(s_predict)
#        Y_SOL.append(y_sol)
#    POLY = np.polyfit(X_plot1, Y_SOL,3)
#    x_sol = root(funcSolPoly, 0.)
##    print("x_sol= "+str(x_sol.x[0]))
#    X_plot2.append(x_sol.x[0])
#
#plotFunction(X_plot1, X_plot2, 'r-', 'Kriging model')
    


# ******************************* ITERATION **********************************#

for i in range(maxIter):    
    S1 = NEW_S1
    S2 = NEW_S2
    
    S_POS = []
    S_NEG = []
    
#    sigma = 1.
    s2_min = S2[0]
    ev_s2min = k.predict(s2_min)
    sigma_ini = (k.predict(s2_min))**0.5
    u_min = abs(ev_s2min)/sigma_ini
    #print(u_min)
    
    if u_min < 5.:
        print("\n****  iteration number: "+str(i+1)+"/"+str(maxIter)+"  ****")
        for j in range(1, len(S2), 1):
            s2 = S2[j]
            #s2min = S2[ind_s2min]
            ev_s2 = k.predict(s2)
            sigma = (k.predict_var(s2))**0.5
#            print(sigma)
            #print("sigma= "+str(sigma_ini))
            u = abs(ev_s2)/sigma
#            print(u)
            if u < u_min:
                #ev_s2min = ev_s2
                u_min = u
                ind_s2min = j
        '''if umin < 5.:
        
            print("\n****  iteration number: "+str(i+1)+"/"+str(maxIter)+"  ****")

            # calculate u
            ind_s2min = 0
            s2min = S2[ind_s2min]
            ev_s2min = k.predict(s2min)
            sigma = 1.
            print("sigma= "+str(sigma))
            umin = abs(ev_s2min)/sigma'''
        '''for j in range(1,len(S2),1):              
            s2 = S2[j]
            ev_s2 = k.predict(s2)
            u = abs(ev_s2)/sigma
            if u < umin:
                ev_s2min = ev_s2
                ind_s2min = j
                umin = u'''
#        print("ind_s2min= "+str(ind_s2min))
#        print("umin= "+str(umin))
        add_s = S2[ind_s2min]
        NEW_S1 = np.append(S1,[add_s],axis=0)
        print("number of construction ponits: "+str(len(NEW_S1)))
        NEW_S2 = np.delete(S2,ind_s2min,axis=0)
#        print("add_s= "+str(add_s))
        y1 = volkersen(add_s)
        Y1.append(y1)
        k.addPoint(add_s,y1)
        k.train()
        #k.plot(show=True)

        # calculate failure probability
        for s in S0:
            ev_s = k.predict(s)

            if ev_s<=0:
                S_NEG.append(s)
            else:
                S_POS.append(s)

        pf = float(len(S_NEG))/float(len(S0))
        print("failure probability= "+str(pf))
        
        # plot figures
        plt.figure(figsize=(10,10))
        plotSamples(S_POS, 10, 'r')
        plotSamples(S_NEG, 10, 'g')
        plotSamples(NEW_S1, 80, 'b')
        plotSamples([add_s], 150, 'y')
        X_plot1 = np.arange(-3., 3.1, 0.1)
        X_plot2 = []
        nPlot = len(X_plot1)
        
        for i in range(nPlot):
            Y_SOL = []
            for j in range(nPlot):
                s_predict = [X_plot1[i],X_plot1[j]]
                y_sol = k.predict(s_predict)
                Y_SOL.append(y_sol)
            POLY = np.polyfit(X_plot1, Y_SOL,3)  # 系数
            #print(POLY)
            x_sol = root(funcSolPoly, 0.)
#            print("x_sol= "+str(x_sol.x[0]))
            X_plot2.append(x_sol.x[0])
        plotFunction(X_plot1, X_plot2, 'c-', 'Kriging model')

    else:
        break    
elapsed = (time.clock() - start)
print("Time used: ", elapsed)

