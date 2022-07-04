import numpy as np
import random
import time
from statistics import variance, mean
import matplotlib.pyplot as plt

normal='real'

def W(gamma,alpha,beta,normal):
    if normal=='real':
       M=np.random.normal(gamma, 1, size=(alpha, beta))
       return np.dot(M,M.T)
    if normal=='complex':
       x=random.random()
       y=random.random()
       M1=np.random.normal(gamma*np.cos(x), np.cos(y), size=(alpha, beta))
       M2=1j*np.random.normal(gamma*np.sin(x), np.sin(y), size=(alpha, beta))
       M=M1+M2
       return np.dot(M,M.conj().T)

def eta(gamma,alpha,beta,num):
    data=[]
    for i in range(num):
        rho=W(gamma,alpha,beta,normal)/beta
        Eig=np.linalg.eigvals(rho)
        Eig=np.real(Eig)
        Eig=np.sort(Eig)
        data.append(Eig[-1])
    return mean(data)

def scale_global(alpha,beta,num,s):
    Z=[x for x in np.linspace(-2,2,s)]
    data=[]
    for gamma in Z:
        data.append(eta(gamma,alpha,beta,num))
    return Z,data

def scale_zoom(alpha,beta,num,s):
    Z=[x for x in np.linspace(-0.2,0.2,s)]
    data=[]
    for gamma in Z:
        data.append(eta(gamma,alpha,beta,num))
    return Z,data

start_time = time.time()

x,y=scale_global(100,100,500,50)
xbis,ybis=scale_zoom(100,100,500,50)

xlin=np.linspace(-2,2,1000)
ylin=100*xlin**2

xlinbis=np.linspace(-0.2,0.2,1000)
ylinbis=100*xlinbis**2

fig, (ax1, ax2) = plt.subplots(2,figsize=(6,8.25))

ax1.plot(xlin,ylin,'-',color='blueviolet',label=''r'$\alpha\gamma^2$')
ax1.plot(x,y,color='forestgreen',linestyle='dashed',label=''r'$\lambda_0$')
ax1.set_xlabel(''r'$\gamma$')
ax1.set_ylabel('eigenvalues')
ax1.set_title('Global')

ax2.plot(xlinbis,ylinbis,'-',color='blueviolet')
ax2.plot(xbis,ybis,color='forestgreen',linestyle='dashed')
ax2.set_xlabel(''r'$\gamma$')
ax2.set_ylabel('eigenvalues')
ax2.set_title('Zoom')

ax1.legend(bbox_to_anchor=(1, 1))
plt.tight_layout() 
plt.savefig('FIG_2.pdf')

print("--- %s seconds ---" % (time.time() - start_time))
