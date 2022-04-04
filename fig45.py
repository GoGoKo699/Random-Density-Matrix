import numpy as np
import random
import time
import matplotlib.pyplot as plt

normal='real'

def RM(alpha,normal):
    if normal=='real':
       l0=random.uniform(0,1)
       sigma=np.sqrt((1-l0)/alpha)
       gamma=random.choice([-1,1])*np.sqrt(l0/alpha)
       M=np.random.normal(gamma, sigma, size=(alpha,alpha))
       rho=np.dot(M,M.T)/alpha
       rho=rho/np.trace(rho)
       Eig=np.linalg.eigvals(rho)
       Eig=np.real(Eig)
       Eig=np.sort(Eig)
       return Eig,Eig[-1],Eig[-2]
    if normal=='complex':
       l0=random.uniform(0,1)
       sigma=np.sqrt((1-l0)/alpha)
       gamma=random.choice([-1,1])*np.sqrt(l0/alpha)
       M1=np.random.normal(gamma, sigma, size=(alpha,alpha))
       M2=1j*np.random.normal(gamma, sigma, size=(alpha,alpha))
       M=(M1+M2)/2
       rho=np.dot(M,M.conj().T)/alpha
       rho=rho/np.trace(rho)
       Eig=np.linalg.eigvals(rho)
       Eig=np.real(Eig)
       Eig=np.sort(Eig)
       return Eig,Eig[-1],Eig[-2]

def Renyi(Eig,d):
    SUM=0
    for p in Eig:
        SUM=SUM+p**d
    SUM=np.log(SUM)
    SUM=SUM/(1-d)
    return SUM

def f(a,l):
    return -1/2*l*(4*np.log(2)-1)-l*np.log(l)+(l-1)*np.log(-4*(l-1)/a)+2*np.log(2)-1/2

def f2(a,l):
    return -np.log(((a + 2)*l**2-4*l+2)/a)

def f_gap(a,l):
    return np.log(l)-np.log(4*(1-l)/a)

start_time = time.time()

alpha=128
samples=100
EIG=[]
MAX=[]
SEC=[]
for i in range(samples):
    Eig,Max,Sec=RM(alpha,normal)
    EIG.append(Eig)
    MAX.append(Max)
    SEC.append(Sec)

MAX_1=[]
ENT_1=[]
for i in range(samples):
    Ent=Renyi(EIG[i],1.0001)
    ENT_1.append(Ent)
    MAX_1.append(MAX[i])

MAX_2=[]
ENT_2=[]
for i in range(samples):
    Ent=Renyi(EIG[i],2)
    ENT_2.append(Ent)
    MAX_2.append(MAX[i])

MAX_inf=[]
ENT_inf=[]
for i in range(samples):
    Ent=Renyi(EIG[i],100)
    ENT_inf.append(Ent)
    MAX_inf.append(MAX[i])

MAX_gap=[]
ENT_gap=[]
for i in range(samples):
    MAX_gap.append(MAX[i])
    ENT_gap.append(np.log(MAX[i])-np.log(SEC[i]))

l=np.linspace(0.02,0.999,1000)

plt.figure(1,figsize=(8,6))
plt.axhline(y=np.log(alpha)-0.5,color='gray',linestyle='dashed') 
plt.plot(l,f(alpha,l),'r-',linewidth=1)
plt.plot(l,f2(alpha,l),'m-',linewidth=1)
plt.plot(l,-np.log(l),'b-',linewidth=1)  
plt.plot(MAX_1,ENT_1,'r+',markersize=7,label=''r'$d\rightarrow 1$')
plt.plot(MAX_2,ENT_2,'m+',markersize=7,label=''r'$d=2$')
plt.plot(MAX_inf,ENT_inf,'b+',markersize=7,label=''r'$d\rightarrow\infty$')
plt.xlim(-0.05,1.05)
plt.xlabel(''r'$\lambda_0$')
plt.ylabel(''r'R$\acute{e}$nyi entropy')
plt.title(''r'$\alpha=\beta=128$')
plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout() 
plt.savefig('FIG_4.pdf')

plt.figure(2,figsize=(8,6))
plt.plot(l,f_gap(alpha,l),'k-',linewidth=1)
plt.plot(MAX_gap,ENT_gap,'k+',markersize=7,label=''r'$\delta\xi$')
plt.xlim(-0.05,1.05)
plt.xlabel(''r'$\lambda_0$')
plt.ylabel('Entanglement Gap')
plt.title(''r'$\alpha=\beta=128$')
plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout() 
plt.savefig('FIG_5.pdf')

print("--- %s seconds ---" % (time.time() - start_time))
