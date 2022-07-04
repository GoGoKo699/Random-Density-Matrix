import numpy as np
import random
import time
import matplotlib.pyplot as plt

normal='real'

def RM(alpha,beta,normal):
    if normal=='real':
       l0=random.uniform(0,1)
       sigma=np.sqrt((1-l0)/alpha)
       gamma=np.sqrt(l0/alpha)
       M=random.choice([-1,1])*np.random.normal(gamma, sigma, size=(alpha,beta))
       rho=np.dot(M,M.T)/alpha
       rho=rho/np.trace(rho)
       Eig=np.linalg.eigvals(rho)
       Eig=np.real(Eig)
       Eig=np.sort(Eig)
       return Eig
    if normal=='complex':
       l0=random.uniform(0,1)
       sigma=np.sqrt((1-l0)/alpha)
       gamma=random.choice([-1,1])*np.sqrt(l0/alpha)
       M1=np.random.normal(gamma, sigma, size=(alpha,beta))
       M2=1j*np.random.normal(gamma, sigma, size=(alpha,beta))
       M=(M1+M2)/2
       rho=np.dot(M,M.conj().T)/alpha
       rho=rho/np.trace(rho)
       Eig=np.linalg.eigvals(rho)
       Eig=np.real(Eig)
       Eig=np.sort(Eig)
       return Eig

def Svon(Eig):
    S=0
    for x in Eig:
        if 1e-15> x.real > -1e-15: 
           S=S
        else: 
           S=S+(abs(x))*np.log(abs(x))
    return -S

def f(a,b,l):
    result=4*a*b*l*np.log(l)+ (8*a*b*np.log(2) - a**2 - b**2)*(l-1)
    result=result-4*(a*b*l - a*b)*np.log(-2*((a + b)*l - (b-a)*(1-l) - a - b)/(a*b))
    result=result- (a+b)*(b-a)*(1-l)
    return -1/4*result/(a*b)

def f_inf(a,l):
    result=(-2*np.log(2)-np.log(a)-np.log(l))*l+(l-1)*np.log(4-4*l)+2*np.log(2)+np.log(a)
    return result

alpha=128

x_128=[]
y_128=[]
for i in range(100):
    Eig=RM(alpha,128,normal)
    x_128.append(Eig[-1])
    y_128.append(Svon(Eig))

x_512=[]
y_512=[]
for i in range(100):
    Eig=RM(alpha,256,normal)
    x_512.append(Eig[-1])
    y_512.append(Svon(Eig))

x_2048=[]
y_2048=[]
for i in range(100):
    Eig=RM(alpha,8192,normal)
    x_2048.append(Eig[-1])
    y_2048.append(Svon(Eig))

l=np.linspace(0.02,0.999,1000)

plt.figure(1,figsize=(6,4.5))
plt.axhline(y=np.log(alpha)-0.5,color='gray',linestyle='dashed') 
plt.plot(l,f(alpha,128,l),'-',color='skyblue',linewidth=1)
plt.plot(l,f(alpha,256,l),'-',color='forestgreen',linewidth=1)
plt.plot(l,f_inf(alpha,l),'-',color='blueviolet',linewidth=1)  
plt.plot(x_128,y_128,'+',color='skyblue',markersize=5,label=''r'$\beta=128$')
plt.plot(x_512,y_512,'+',color='forestgreen',markersize=5,label=''r'$\beta=256$')
plt.plot(x_2048,y_2048,'+',color='blueviolet',markersize=5,label=''r'$\beta\rightarrow\infty$')
plt.xlim(-0.05,1.05)
plt.xlabel(''r'$\lambda_0$')
plt.ylabel('Shannon entropy')
plt.title(''r'$\alpha=128$')
plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout() 
plt.savefig('FIG_3.pdf')
