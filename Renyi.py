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
samples=30
EIG=[]
MAX=[]

for i in range(samples):
    Eig,Max=RM(alpha,normal)
    EIG.append(Eig)
    MAX.append(Max)
    
MAX_0=[]
ENT_0=[]
for i in range(samples):
    Ent=Renyi(EIG[i],0)
    ENT_0.append(Ent)
    MAX_0.append(MAX[i])

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

x=np.linspace(0.000001,0.999999,1000)

fig=plt.figure(1,figsize=(4.5,5))
ax = fig.add_subplot()

plt.fill_between(x, f(alpha,x),color='skyblue',alpha=0.3)
plt.fill_between(x,-x*np.log(x)-(1-x)*np.log(1-x), color='white')
plt.fill_between(x, f2(alpha,x),color='forestgreen',alpha=0.3)
plt.fill_between(x, -np.log(x**2+(1-x)**2), color='white')
plt.fill_between(x,-np.log(x), color='white')
plt.fill_between(x, 0, np.log(alpha), where=(x<4/alpha), color='white')
plt.axvline(x=4/alpha,ymin=(np.log(alpha)-np.log(4))/np.log(alpha),ymax=(np.log(alpha)-1/2)/np.log(alpha),color='gray',linestyle='dotted')

x=np.linspace(0.5,0.999999,1000)
plt.plot(x,-x*np.log(x)-(1-x)*np.log(1-x),color='gray',linestyle='dashdot')
plt.plot(x,-np.log(x**2+(1-x)**2),color='gray',linestyle='dashdot')

x=np.linspace(4/alpha,0.999999,1000)
plt.plot(x,f(alpha,x),'-',color='skyblue',linewidth=1)
plt.plot(x,f2(alpha,x),'-',color='forestgreen',linewidth=1)
plt.plot(x,-np.log(x),color='gray',linestyle='dashed')
plt.plot(MAX_0,ENT_0,'+',color='gray',markersize=5,label=''r'$d=0$')
plt.plot(MAX_1,ENT_1,'+',color='skyblue',markersize=5,label=''r'$d\rightarrow 1$')
plt.plot(MAX_2,ENT_2,'+',color='forestgreen',markersize=5,label=''r'$d=2$')
plt.plot(MAX_inf,ENT_inf,'+',color='blueviolet',markersize=5,label=''r'$d=100$')
plt.xlim(0,1)
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0])
plt.xlabel(''r'$\lambda_0$')
plt.ylabel('entropy')
plt.ylim(0,np.log(alpha))
plt.title(''r'$\alpha=\beta=128$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fancybox=True, shadow=True, ncol=5, fontsize= 9)
ax.set_box_aspect(1)
plt.tight_layout()
#FIG. 21
plt.savefig('Renyi.pdf')

print("--- %s seconds ---" % (time.time() - start_time))
