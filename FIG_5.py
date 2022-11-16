import numpy as np
import random
import time
import matplotlib.pyplot as plt

start_time = time.time()

normal='real'

def RM(alpha,normal):
    if normal=='real':
       l0=random.uniform(0,1)
       sigma=np.sqrt((1-l0)/alpha)
       gamma=random.choice([-1,1])*np.sqrt(l0/alpha)
       M=np.random.normal(gamma, sigma, size=(alpha,alpha))
       rho=np.dot(M,M.T)
       rho=rho/np.trace(rho)
       Eig=np.linalg.eigvals(rho)
       Eig=np.real(Eig)
       Eig=np.sort(Eig)
       return Eig,Eig[-1]
    if normal=='complex':
       l0=random.uniform(0,1)
       sigma=np.sqrt((1-l0)/alpha)
       gamma=random.choice([-1,1])*np.sqrt(l0/alpha)
       M1=np.random.normal(gamma, sigma, size=(alpha,alpha))
       M2=1j*np.random.normal(gamma, sigma, size=(alpha,alpha))
       M=(M1+M2)/2
       rho=np.dot(M,M.conj().T)
       rho=rho/np.trace(rho)
       Eig=np.linalg.eigvals(rho)
       Eig=np.real(Eig)
       Eig=np.sort(Eig)
       return Eig,Eig[-1]

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

for i in range(samples):
    Eig,Max=RM(alpha,normal)
    EIG.append(Eig)
    MAX.append(Max)

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

l=np.linspace(0.02,0.999,1000)

plt.figure(1,figsize=(4.5,4.5))
plt.axhline(y=np.log(alpha)-0.5,color='gray',linestyle='dashed') 
plt.plot(l,f(alpha,l),'-',color='skyblue',linewidth=1)
plt.plot(l,f2(alpha,l),'-',color='forestgreen',linewidth=1)
plt.plot(l,-np.log(l),'-',color='blueviolet',linewidth=1)  
plt.plot(MAX_1,ENT_1,'+',color='skyblue',markersize=5,label=''r'$d\rightarrow 1$')
plt.plot(MAX_2,ENT_2,'+',color='forestgreen',markersize=5,label=''r'$d=2$')
plt.plot(MAX_inf,ENT_inf,'+',color='blueviolet',markersize=5,label=''r'$d\rightarrow\infty$')
plt.xlim(-0.05,1.05)
plt.xlabel(''r'$\lambda_0$')
plt.ylabel('entropy')
plt.title(''r'$\alpha=\beta=128$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
plt.tight_layout() 
plt.savefig('FIG_5.pdf')

print("--- %s seconds ---" % (time.time() - start_time))
