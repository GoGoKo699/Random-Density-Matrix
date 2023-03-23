import numpy as np
import random
import time
import matplotlib.pyplot as plt

start_time = time.time()

normal='real'

def RM(alpha,beta,normal):
    if normal=='real':
       l0=random.uniform(0,1)
       sigma=np.sqrt((1-l0)/alpha)
       gamma=np.sqrt(l0/alpha)
       M=random.choice([-1,1])*np.random.normal(gamma, sigma, size=(alpha,beta))
       rho=np.dot(M,M.T)
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
       rho=np.dot(M,M.conj().T)
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
    result=(1-l)*(np.log(a)-np.log(1-l)-a/(2*b))-l*np.log(l)
    return result

def f_inf(a,l):
    result=(-2*np.log(2)-np.log(a)-np.log(l))*l+(l-1)*np.log(4-4*l)+2*np.log(2)+np.log(a)
    return result

alpha=128

x_128=[]
y_128=[]
for i in range(50):
    Eig=RM(alpha,128,normal)
    x_128.append(Eig[-1])
    y_128.append(Svon(Eig))

x_512=[]
y_512=[]
for i in range(50):
    Eig=RM(alpha,256,normal)
    x_512.append(Eig[-1])
    y_512.append(Svon(Eig))

x_inf=[]
y_inf=[]
for i in range(50):
    Eig=RM(alpha,8192,normal)
    x_inf.append(Eig[-1])
    y_inf.append(Svon(Eig))

x=np.linspace(0.000001,0.999999,1000)

fig=plt.figure(1,figsize=(4.5,5))
ax = fig.add_subplot()

plt.fill_between(x, f(alpha,128,x),color='skyblue',alpha=0.3)
plt.fill_between(x, f(alpha,128,x),f(alpha,256,x),color='forestgreen',alpha=0.3)
plt.fill_between(x, f(alpha,256,x),f(alpha,8192,x),color='blueviolet',alpha=0.3)

plt.plot(x,-x*np.log(x)-(1-x)*np.log(1-x),color='gray',linestyle='dotted')
plt.plot(x,-np.log(x),color='gray',linestyle='dashed')
plt.plot(x,-x*np.log(x)-(1-x)*np.log((1-x)/(alpha)),color='gray',linestyle='dashdot')

plt.fill_between(x, -x*np.log(x)-(1-x)*np.log(1-x),color='silver')
plt.fill_between(x, -x*np.log(x)-(1-x)*np.log((1-x)/(alpha)),np.log(alpha),color='silver')
plt.fill_between(x, -np.log(x),color='silver')

x=np.linspace((1+np.sqrt(alpha/128))**2/alpha,0.999999,1000)
plt.plot(x,f(alpha,128,x),'-', color='skyblue',linewidth=1)
x=np.linspace((1+np.sqrt(alpha/256))**2/alpha,0.999999,1000)
plt.plot(x,f(alpha,256,x),'-', color='forestgreen',linewidth=1)
plt.plot(x_128,y_128,'+',color='skyblue',markersize=5,label=''r'$\beta=128$')
plt.plot(x_512,y_512,'x',color='forestgreen',markersize=5,label=''r'$\beta=256$')
plt.plot(x_inf,y_inf,'*',color='blueviolet',markersize=5,label=''r'$\beta=8192$')
plt.xlim(0,1)
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0])
plt.ylim(0,np.log(alpha))
plt.xlabel(''r'$\lambda_0$')
plt.ylabel('entropy')
plt.title(''r'$\alpha=128$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fancybox=True, shadow=True, ncol=5)
ax.set_box_aspect(1)
plt.tight_layout()
#FIG. 11
plt.savefig('alphabeta.pdf')

print("--- %s seconds ---" % (time.time() - start_time))
