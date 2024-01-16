import numpy as np
import random
import time
from statistics import variance, mean
import matplotlib.pyplot as plt

start_time = time.time()

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
    x_dot=[]
    y_dot=[]
    for i in range(num):
        rho=W(gamma,alpha,beta,normal)/beta
        Eig=np.linalg.eigvals(rho)
        Eig=np.real(Eig)
        Eig=np.sort(Eig)
        x_dot.append(gamma)
        y_dot.append(Eig[-1])
    M=mean(y_dot)
    return x_dot,y_dot,M

def scale_global(alpha,beta,num,s):
    X=[x for x in np.linspace(-1,1,s)]
    X_dot=[]
    Y_dot=[]
    data=[]
    for gamma in X:
        x_dot,y_dot,M=eta(gamma,alpha,beta,num)
        X_dot.extend(x_dot)
        Y_dot.extend(y_dot)
        data.append(M)
    return X_dot,Y_dot,X,data
    
def scale_zoom(alpha,beta,num,s):
    X=[x for x in np.linspace(-0.2,0.2,s)]
    X_dot=[]
    Y_dot=[]
    data=[]
    for gamma in X:
        x_dot,y_dot,M=eta(gamma,alpha,beta,num)
        X_dot.extend(x_dot)
        Y_dot.extend(y_dot)
        data.append(M)
    return X_dot,Y_dot,X,data
    
alpha=100
beta=200
num=500
s=100

x_dot,y_dot,x,y=scale_global(alpha,beta,num,s)
xbis_dot,ybis_dot,xbis,ybis=scale_zoom(alpha,beta,num,s)

xlin=np.linspace(-1,1,1000)
ylin=100*xlin**2

xlinbis=np.linspace(-0.2,0.2,1000)
ylinbis=100*xlinbis**2

fig, (ax1, ax2) = plt.subplots(2,figsize=(4,7.5))

ax1.axhline(y=(1+np.sqrt(2)/2)**2,color='gray',linestyle='dotted') 
ax1.plot(x_dot,y_dot,'o',color='skyblue',alpha=0.05,markersize=1)
ax1.axhline(y=(1+np.sqrt(2)/2)**2,color='gray',linestyle='dotted') 
ax1.plot(xlin,ylin,'-',color='blueviolet')
ax1.plot(x,y,color='forestgreen',linestyle='dashed')
ax1.set_xlabel(''r'$\gamma$')
ax1.set_ylabel('eigenvalues')
ax1.set_title('global')

ax2.plot(xbis_dot,ybis_dot,'o',color='skyblue',alpha=0.05,markersize=1)
ax2.axhline(y=(1+np.sqrt(2)/2)**2,color='gray',linestyle='dotted')
ax2.plot(xlinbis,ylinbis,'-',color='blueviolet',label=''r'$\alpha\gamma^2$')
ax2.plot(xbis,ybis,color='forestgreen',linestyle='dashed',label=''r'$\lambda_0$')
ax2.set_xlabel(''r'$\gamma$')
ax2.set_ylabel('eigenvalues')
ax2.set_title('zoom')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), fancybox=True, shadow=True, ncol=5)

fig.tight_layout()
#FIG. 10
plt.savefig('mean_MPD.pdf')

print("--- %s seconds ---" % (time.time() - start_time))
