# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 16:24:27 2019

@author: Shoqi WU
La plupart de programme retrait dans rbaachaoui
combiant trois partie de l'analyse pour lancer le calcul automatiquement
corriger le formule de G dans le fichier INP pour E
"""


import numpy as np
from numpy import cosh, sinh

import pyKriging  
from pyKriging.krige import kriging

import matplotlib.pyplot as plt
from scipy.optimize import minimize, root

import random
import os

import pandas as pd
import csv
#****************************************************************************** Lancement Calcul***************************************************************#

# ******************************** fonctions pour l'INP ******************************** #

def readCoordNode(oldInp,firstLine,lastLine):
    """ fonction qui recherche les coordonnes entre firstline et lastline"""
    oldInp.seek(0)
    Data=[]
    for i,u in enumerate(oldInp):
        if (i>=firstLine) and (i<=lastLine):
            try:
                Data.append([float(x) for x in u.split(',')])
            except:
                pass
    oldInp.seek(0)
    return np.array(Data)

def coord_noeud(node):
    if node -int(node)==0:
        coord = str(int(node))+'.'
    else:
        coord = str(node)
    return coord

def format_node(Node):
    texte = ''
    for ii_node in range(0,len(Node[:,0])):
        coorx,coory,coorz = coord_noeud(Node[ii_node,1]),coord_noeud(Node[ii_node,2]),coord_noeud(Node[ii_node,3])
        texte += '{0:>7s}'.format(str(int(Node[ii_node,0]))) + ',' + '{0:>13s}'.format(coorx) + ',' + '{0:>13s}'.format(coory) + ',' + '{0:>13s}'.format(coorz) + '\n'
    return texte

def formatnode(Node):
    texte = ''
    E,mu = coord_noeud(Node[:,0]),coord_noeud(Node[:,1])
    texte += '{0:>5s}'.format(E) + ',' + '{0:>4s}'.format(mu) +  '\n'
    return texte


def replaceText(oldText,newText,oldInpText):
    """ fonction qui remplace le texte"""
    newInpText=oldInpText.replace(oldText,newText)
    return newInpText


def occurLine(oldInp,occurence):
    """ fonction qui recherche le texte"""
    oldInp.seek(0)
    for j,u in enumerate(oldInp):
        if (occurence==u):
            break
    return j

def occurLineall(oldInp,firstoordLineName,lastoordLineName):
    """ fonction qui remplace la 1er ligne et la derniere ligne"""
    oldInp.seek(0)
    firstoordLine,lastoordLine=0.,0.
    for j,u in enumerate(oldInp):
        if (firstoordLineName==u):
            print(firstoordLineName,j)
            firstoordLine=j
        if  (lastoordLineName==u) and (firstoordLine > 0):
            print(lastoordLineName,j)
            lastoordLine=j
            break
    return firstoordLine+2,lastoordLine   
    
# ******************************** fonctions pour le fichier .dat ******************************** #

def readCoordNodeDAT(oldDat,firstLine,lastLine):
    """ fonction qui recherche les coordonnes entre firstline et lastline"""
    oldDat.seek(0)
    Data=[]
    for i,u in enumerate(oldDat):
        if (i>=firstLine) and (i<=lastLine):
            try:
                Data.append([float(x) for x in u.split()])
            except:
                pass
    oldDat.seek(0)
    return np.array(Data)
    
def coord_noeudDAT(node):
    if node -int(node)==0:
        coord = str(int(node))+'.'
    else:
        coord = str(node)
    return coord

def replaceTextDAT(oldText,newText,oldDatText):
    """ fonction qui remplace le texte"""
    newDatText=oldDatText.replace(oldText,newText)
    return newDatText

def format_nodeDAT(A):
    texte = ''
    for ii_A in range(0,len(A[:,0])):
       element,note,s13 = coord_noeudDAT(A[ii_A,0]),coord_noeudDAT(A[ii_A,1]),coord_noeudDAT(A[ii_A,2])
       texte += '{0:>9s}'.format(element) + '{0:>3s}'.format(note)+ '{0:>10s}'.format(s13) + '\n'
    return texte
    
def occurLineallDAT(oldDat,firstoordLineName,lastoordLineName):
    """ fonction qui remplace la 1er liegne et la derniere ligne"""
    oldDat.seek(0)
    firstoordLine,lastoordLine=0.,0.
    for j,u in enumerate(oldDat):
        if (firstoordLineName==u):
            print(firstoordLineName,j)
            firstoordLine=j
        if  (lastoordLineName==u) and (firstoordLine > 0):
            print(lastoordLineName,j)
            lastoordLine=j
            break
    return firstoordLine+20,lastoordLine-8

# ******************************** fonction pour lancer le fichier INP sur ABAQUS ******************************** #
def lancement_calcul(ep,G):
    # *** Créer un fichier INP pour chaque couple de valeur ea, Ga *** #
    fidName=mainDir + "assemblage-colle.inp"
    fid=open(fidName,"r")
    newInpName=mainDir + "NewInp.inp"
    newInp=open(newInpName,"w")
    oldInp=fid.read()
    
    
    
    firstCoordLineName="*Part, name=Top-Part\n"
    lastCoordLineName="*Element, type=SC8R\n"
    firstCoordLine, lastCoordLine = occurLineall(fid,firstCoordLineName,lastCoordLineName)
    print("debut : "+ str(firstCoordLine)+ " --- fin :"+ str(lastCoordLine))
    DataNode=readCoordNode(fid,firstCoordLine,lastCoordLine)
    Nodenew = np.copy(DataNode)
#    print(Nodenew)
    Nodenew[:,3] = DataNode[:,3]+ ep- ep0
#    print(Nodenew)
    old_node_texte = format_node(DataNode)
    new_node_texte = format_node(Nodenew)
#    print(new_node_texte)
    newInpText2=replaceText(old_node_texte[0:len(old_node_texte)],new_node_texte[0:len(new_node_texte)],oldInp)
#    print(newInpText2)
    
    
        
    firstLineName2="*Part, name=Cohesive\n"
    lastLineName2="*Element, type=C3D8R\n"
    firstLine2, lastLine2 = occurLineall(fid,firstLineName2,lastLineName2)
    print("debut : "+ str(firstLine2)+ " --- fin :"+ str(lastLine2))
    DataNode2=readCoordNode(fid,firstLine2,lastLine2)
    d1=ep-ep0
    d2=(ep-ep0)*(2./3)
    d3=(ep-ep0)/3
    a = []
    for i in range(len(DataNode2[:,3])):
        if DataNode2[i,3]==1:
            a.append(DataNode2[i,3]+d1)
        elif DataNode2[i,3]==0.666666687:
            a.append(DataNode2[i,3]+d2)
        elif DataNode2[i,3]==0.333333343:
            a.append(DataNode2[i,3]+d3)
        else:
            a.append(DataNode2[i,3])
  
    A = np.array(a)
    Nodenew2 = np.copy(DataNode2)
    Nodenew2[:,3] = A  
    old_node_texte = format_node(DataNode2)
    new_node_texte = format_node(Nodenew2)
    newInpText3=replaceText(old_node_texte[0:len(old_node_texte)],new_node_texte[0:len(new_node_texte)],newInpText2)
    
    
  
    oldText = '10400., 0.3'
    newText = str(2.6*G) + ', 0.3'  #formule G est corrigé E = 2.6*G
    newInpText4=replaceText(oldText,newText,newInpText3) 
    
    
    old_texte='*Output, field, variable=PRESELECT'
    new_texte='*Output, field\n*element output, elset=Set-cisaillement\ns\n*El print, elset=Set-cisaillement\nMises'
    newInpText5=replaceText (old_texte,new_texte,newInpText4) 



    newInp.write(newInpText5)
    fid.close()
    newInp.close() 
    
    
    # *** fichier Bat *** #
    print('epaisseur:',ep,'module_de_cisaillement :',G)
    os.chdir(mainDir)
    os.system(mainDir+"batFile.bat")
    
    
    
    # *** Extraire la contrainte pour chaque couple de valeur ea, Ga *** #

    fidNameDat=mainDir  + "NewInp.dat"
    fidDat=open(fidNameDat,"r")
    newDatName=mainDir + "extract_data"+".csv"
    extract_data=open(newDatName,"w")
    oldDat=fidDat.read()
    
    firstCoordLineName2="                                INCREMENT    20 SUMMARY\n"
    lastCoordLineName2="          THE ANALYSIS HAS BEEN COMPLETED\n"
    firstCoordLine2, lastCoordLine2 = occurLineallDAT(fidDat,firstCoordLineName2,lastCoordLineName2)
    print("debut : "+ str(firstCoordLine2)+ " --- fin :"+ str(lastCoordLine2))
    
    
    DataNode2=readCoordNodeDAT(fidDat,firstCoordLine2,lastCoordLine2-1)
    new_node_texte = format_nodeDAT(DataNode2)
    #
    contrainte=DataNode2[:,2] 
    Y=max(contrainte)-20
    #  
    extract_data.write(new_node_texte)
    fidDat.close()
    extract_data.close()   
    return Y    
  

    
    
#******************************************************************************FIN Lancement Calcul***************************************************************#
   

def samplingMC(nMC):
    X = []
    for i in range(nMC):
        random.seed(a=None)
        x = random.gauss(0,1)
        X.append(x)
    return X

    
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

def format_node_add(A):
    texte = ''  
    E,G = coord_noeud(A[0]),coord_noeud(A[1])
    texte +=str(E)+'\n'+str(G)
    return texte    

def convertCtrReal(X, a, b):
    return a*X+b
    
#def coord_noeud(node):
    if node -int(node)==0:
        coord = str(int(node))+'.'
    else:
        coord = str(node)
    return coord
    
def format_node_mc(A):
    texte = ''
    for ii_A in range(0,len(A[:,0])):
       E,G = coord_noeud(A[ii_A,0]),coord_noeud(A[ii_A,1])
       texte +='{0:>3s}'.format(E)+','+ '{0:>10s}'.format(G) +  '\n'
    return texte

def format_node_indices(A):
    texte = ''
    for ii_A in range(len(A)):
       indice = coord_noeud(A[ii_A])
       texte +='{0:>3s}'.format(indice)+ '\n'
    return texte    


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
    
    
# ------------------------------- Main scipt ---------------------------------# 
# define sampling size
nMC = 5000

# define initial number of individuals to be used to build kriging model 
N1 = 5

# sampling nMC value of e0 and G0
E0 = np.array(samplingMC(nMC))
G0 = np.array(samplingMC(nMC))

# convert to real variation domain
aE, bE = 0.2, 1.
aG, bG = 1000., 4000.
E = convertCtrReal(E0,aE,bE)
G = convertCtrReal(G0,aG,bG)

# reshape inputs
S0 = np.reshape([E0, G0],(nMC,2))



# get initial population (N1 individual)
S1, S2, Indices = getInitialPop(S0, nMC, N1)

mainDir = "C:\\Shaoqi_Stage\\code\modele_final\\resultat_essai_numerique\\"
os.chdir(mainDir)   


newDatName=mainDir + "extractS1"+".csv"
extractS1=open(newDatName,"w")
new_node_texte1 = format_node_mc(S1)
extractS1.write(new_node_texte1)
extractS1.close()  

newDatName=mainDir + "extractS2"+".csv"
extractS2=open(newDatName,"w")
new_node_texte2 = format_node_mc(S2)
extractS2.write(new_node_texte2)
extractS2.close()  

newDatName=mainDir + "extractS0"+".csv"
extractS0=open(newDatName,"w")
new_node_texte0 = format_node_mc(S0)
extractS0.write(new_node_texte0)
extractS0.close()  

newDatName=mainDir + "extractIndices"+".csv"
extractIndices=open(newDatName,"w")
new_node_texte_Indices = format_node_indices(Indices)
extractIndices.write(new_node_texte_Indices)
extractIndices.close()  

E1=S1[:,0]
G1=S1[:,1]
E11 = convertCtrReal(E1,aE,bE)
G11 = convertCtrReal(G1,aG,bG)


Y1 = []
ep0 = 1
fichier = open('resultat.txt','w')
#fichier.write('Y\n')
fichier.close()   
for ep,G in zip(E11,G11):
    Y= lancement_calcul(ep,G) 
    fichier = open('resultat.txt','a')
    fichier.write(str(Y)+'\n') 
    fichier.close()


# -----------------------------------------------enrishissement-----------------------------------------#  
  
#mainDir = "C:\\Shaoqi_Stage\\code\\modele_final\\resultat_essai_numerique\\"
#os.chdir(mainDir)   
    
fidName0=mainDir + "extractS0.csv"
fid0=open(fidName0,"r")
oldS0=fid0.read()
S0=pd.read_csv('extractS0.csv',header=-1)
S0 = S0.values
fid0.close()
    
fidName1=mainDir + "extractS1.csv"
fid1=open(fidName1,"r")
oldS1=fid1.read()
S1=pd.read_csv('extractS1.csv',header=-1)
S1 = S1.values
fid1.close()
    
fidName2=mainDir + "extractS2.csv"
fid2=open(fidName2,"r")
oldS2=fid2.read()
S2=pd.read_csv('extractS2.csv',header=-1)
S2 = S2.values
fid2.close()
    
f = open("resultat.txt","r+")
d = f.readlines()
f.seek(0)
for i in d:
    if i != "":
        f.write(i)
f.truncate()
f.close()
    
fidName3=mainDir + "resultat.txt"
fid3=open(fidName3,"r")
contrainte=mainDir + "Y"
resultat=fid3.read()
Y=np.loadtxt('resultat.txt')  #
fid3.close()
    
    
    
# *********************** BUILDING INITIAL KRIGING MODEL *********************#
#
print(len(S1))
print(len(Y))
k = kriging(S1, Y, name='simple')
    
k.train()




# ********************************** PLOT ************************************#
print("\n****  Initialization  ****")
#
#
#
#umin = 0
#
## calculate failure probability
S_POS = []
S_NEG = []
for s in S0:
    ev_s = k.predict(s)
    if ev_s<=0:
        S_NEG.append(s)
    else:
        S_POS.append(s)
#
pf = float(len(S_NEG))/float(len(S0))
print("failure probability= "+str(pf))
#
## plot figures
plt.figure(figsize=(10,10))
plotSamples(S_POS, 10, 'r')
plotSamples(S_NEG, 10, 'g')
plotSamples(S1, 100, 'b')

#X_plot1 = np.arange(-3., 3.1, 0.1)
#X_plot2 = []
#nPlot = len(X_plot1)

#for i in range(nPlot):
#    Y_SOL = []
#    for j in range(nPlot):
#        s_predict = [X_plot1[i],X_plot1[j]]
#        y_sol = k.predict(s_predict)
#        Y_SOL.append(y_sol)
#    POLY = np.polyfit(X_plot1, Y_SOL,3)
#    x_sol = root(funcSolPoly, 0.)
#    X_plot2.append(x_sol.x[0])
#
#plotFunction(X_plot1, X_plot2, 'r-', 'Kriging model')
#plt.show()
#plt.draw()    
#    


# ******************************* ITERATION **********************************#
ITER = 15       
for fois in range(1,ITER+1,1):  
    print("\n****  nouvelle iteration " + str(fois) +":   ****")    
    

    S_POS = []
    S_NEG = []
    
    

# calculate u

    ind_s2min = 0
    s2min = S2[ind_s2min]
    ev_s2min = k.predict(s2min)
    sigma_ini = (k.predict_var(s2min))**0.5
    umin = abs(ev_s2min)/sigma_ini
    if umin < 5:
        
        for j in range(1,len(S2),1):            
            s2 = S2[j]
            ev_s2 = k.predict(s2)
            sigma = (k.predict_var(s2))**0.5
            u = abs(ev_s2)/sigma
            if u < umin:
                ev_s2min = ev_s2
                ind_s2min = j
                umin = u
    add_s = S2[ind_s2min]
    S1 = np.append(S1,[add_s],axis=0)
    S2 = np.delete(S2,ind_s2min,axis=0)  

# ******************************* create new S1 and S2 **********************************#

    newDatName=mainDir + "extractS1"+".csv"
    extractS1=open(newDatName,"w")
    new_node_texte1 = format_node_mc(S1)
    extractS1.write(new_node_texte1)
    extractS1.close()  
    
    newDatName=mainDir + "extractS2"+".csv"
    extractS2=open(newDatName,"w")
    new_node_texte2 = format_node_mc(S2)
    extractS2.write(new_node_texte2)
    extractS2.close()  


    fidName=mainDir + "add_s1.txt"
    add_s1=open(fidName,"w")
    new_node_texte_add = format_node_add(add_s)
    add_s1.write(new_node_texte_add)
    add_s1.close()  

# --------------------------------------------lance1pt-------------------------------#
#    mainDir = "C:\\Shaoqi_Stage\\code\\modele_final\\resultat_essai_numerique\\"
#    os.chdir(mainDir)
    
    fidName=mainDir + "add_s1.txt"
    fid=open(fidName,"r")
    add=mainDir + "s"
    olds=fid.read()
    s=np.loadtxt('add_s1.txt')
    fid.close()
    
    ep0 = 1
    e_add=s[0]
    G_add=s[1]
    # convert to real variation domain
    aE, bE = 0.2, 1.
    aG, bG = 1000., 4000.
    E = convertCtrReal(e_add,aE,bE)
    G = convertCtrReal(G_add,aG,bG)
    
    add_Y= lancement_calcul(E,G) 
    fichier = open('resultat.txt','a')
    fichier.write(str(add_Y)+'\n') 
    Y = np.loadtxt('resultat.txt')
    fichier.close()

    k.addPoint(s,add_Y)
    k.train()
    
# calculate failure probability

    for ss in S0:
        ev_s = k.predict(ss)
    
        if ev_s<=0:
            S_NEG.append(ss)
        else:
            S_POS.append(ss)
    
    pf = float(len(S_NEG))/float(len(S0))
    print("failure probability= "+str(pf))

# plot figures

    plt.figure(figsize=(10,10))
    plotSamples(S_POS, 10, 'r')
    plotSamples(S_NEG, 10, 'g')
    plotSamples(S1, 80, 'b')
    plotSamples([add_s], 150, 'y')
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
    #
    #    X_plot2.append(x_sol.x[0])
    #plotFunction(X_plot1, X_plot2, 'r-', 'Kriging model')
    #plt.show()
    #plt.draw()




