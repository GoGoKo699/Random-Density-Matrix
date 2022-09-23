import numpy as np
import random
import time
import matplotlib.pyplot as plt

normal='real'

def RM(alpha,beta,normal):
    if normal=='real':
       l0=random.uniform(0,1)
       sigma=np.sqrt((1-l0)/alpha)
       gamma=random.choice([-1,1])*np.sqrt(l0/alpha)
       M=np.random.normal(gamma, sigma, size=(alpha,beta))
       rho=np.dot(M,M.T)
       rho=rho/np.trace(rho)
       Eig=np.linalg.eigvals(rho)
       Eig=np.real(Eig)
       Eig=np.sort(Eig)
       return Eig[-1],Eig[-2]
    if normal=='complex':
       l0=random.uniform(0,1)
       sigma=np.sqrt((1-l0)/alpha)
       gamma=random.choice([-1,1])*np.sqrt(l0/alpha)
       M1=np.random.normal(gamma, sigma, size=(alpha,beta))
       M2=1j*np.random.normal(gamma, sigma, size=(alpha,beta))
       M=(M1+M2)/2
       rho=np.dot(M,M.conj().T)
       rho=rho/np.trace(rho)
       Eig=np.linalg.eigvals(rho)
       Eig=np.real(Eig)
       Eig=np.sort(Eig)
       return Eig[-1],Eig[-2]

def f_gap(alpha,beta,l):
    return np.log(l)-np.log(((1-l)/alpha)*(1+np.sqrt(alpha/beta))**2)
    
def f_gap_inf(alpha,l):
    return np.log(l)-np.log((1-l)/alpha)

start_time = time.time()

alpha=128
samples=100

MAX_128=[]
GAP_128=[]
for i in range(samples):
    Max,Sec=RM(alpha,128,normal)
    MAX_128.append(Max)
    GAP_128.append(np.log(Max)-np.log(Sec))
    
MAX_256=[]
GAP_256=[]
for i in range(samples):
    Max,Sec=RM(alpha,256,normal)
    MAX_256.append(Max)
    GAP_256.append(np.log(Max)-np.log(Sec))
    
MAX_inf=[]
GAP_inf=[]
for i in range(samples):
    Max,Sec=RM(alpha,2**16,normal)
    MAX_inf.append(Max)
    GAP_inf.append(np.log(Max)-np.log(Sec))


l=np.linspace(0.02,0.999,1000)

plt.figure(2,figsize=(6,4.5))
plt.plot(l,f_gap(alpha,128,l),'-',color='skyblue',linewidth=1)
plt.plot(l,f_gap(alpha,256,l),'-',color='forestgreen',linewidth=1)
plt.plot(l,f_gap_inf(alpha,l),'-',color='blueviolet',linewidth=1)  
plt.plot(MAX_128,GAP_128,'+',color='skyblue',markersize=5,label=''r'$\beta=128$')
plt.plot(MAX_256,GAP_256,'+',color='forestgreen',markersize=5,label=''r'$\beta=256$')
plt.plot(MAX_inf,GAP_inf,'+',color='blueviolet',markersize=5,label=''r'$\beta\rightarrow\infty$')
plt.xlim(-0.05,1.05)
plt.xlabel(''r'$\lambda_0$')
plt.ylabel(''r'$\delta\xi$')
plt.title(''r'$\alpha=128$')
plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout() 
plt.savefig('FIG_5.pdf')

print("--- %s seconds ---" % (time.time() - start_time))
