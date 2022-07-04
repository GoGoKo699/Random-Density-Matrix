import numpy as np
import random
import time
from statistics import mean
import matplotlib.pyplot as plt

normal='real'

def W(gamma,alpha,normal):
    if normal=='real':
       M=np.random.normal(gamma, 1, size=(alpha, alpha))
       return M+M.T
    if normal=='complex':
       x=random.random()
       M1=np.random.normal(gamma, np.cos(x), size=(alpha, alpha))
       M2=1j*np.random.normal(2*random.random()*gamma, np.sin(x), size=(alpha, alpha))
       M=M1+M2
       return M+M.conj().T

def eta(gamma,alpha,num):
    data=[]
    for i in range(num):
        rho=W(gamma,alpha,normal)/(2*np.sqrt(alpha))
        Eig=np.linalg.eigvals(rho)
        Eig=np.real(Eig)
        Eig=np.sort(Eig)
        if gamma>0:
           data.append(Eig[-1])
        else: 
           data.append(Eig[0])
    return mean(data)

def scale_global(alpha,num,s):
    Z=[x for x in np.linspace(-2,2,s)]
    data=[]
    for gamma in Z:
        data.append(eta(gamma,alpha,num))
    return Z,data

def scale_zoom(alpha,num,s):
    Z=[x for x in np.linspace(-0.2,0.2,s)]
    data=[]
    for gamma in Z:
        data.append(eta(gamma,alpha,num))
    return Z,data

start_time = time.time()

x,y=scale_global(100,500,50)
xbis,ybis=scale_zoom(100,500,50)

xlin=np.linspace(-2,2,1000)
ylin=100*xlin/(np.sqrt(100))

xlinbis=np.linspace(-0.2,0.2,1000)
ylinbis=100*xlinbis/(np.sqrt(100))

xm=x[:25]
ym=y[:25]
xp=x[25:]
yp=y[25:]

xmbis=xbis[:25]
ymbis=ybis[:25]
xpbis=xbis[25:]
ypbis=ybis[25:]

fig, (ax1, ax2) = plt.subplots(2,figsize=(6,8.25))

ax1.plot(xlin,ylin,'-',color='blueviolet',label=''r'$\frac{\alpha\gamma}{\sqrt{\alpha}}$')
ax1.plot(xm,ym,color='forestgreen',linestyle='dashed',label=''r'$\lambda_0$')
ax1.plot(xp,yp,color='forestgreen',linestyle='dashed')
ax1.set_xlabel(''r'$\gamma$')
ax1.set_ylabel('eigenvalues')
ax1.set_title('Global')

ax2.plot(xlinbis,ylinbis,'-',color='blueviolet')
ax2.plot(xmbis,ymbis,color='forestgreen',linestyle='dashed')
ax2.plot(xpbis,ypbis,color='forestgreen',linestyle='dashed')
ax2.set_xlabel(''r'$\gamma$')
ax2.set_ylabel('eigenvalues')
ax2.set_title('Zoom')

ax1.legend(bbox_to_anchor=(1, 1))
plt.tight_layout() 
plt.savefig('FIG_8.pdf')

print("--- %s seconds ---" % (time.time() - start_time))
