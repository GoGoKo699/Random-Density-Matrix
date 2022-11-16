import numpy as np
import random
import time
import matplotlib.pyplot as plt

start_time = time.time()

normal='real'

def W(alpha,beta,normal):
    if normal=='real':
       M=np.random.normal(0, 1, size=(alpha, beta))
       return np.dot(M,M.T)
    if normal=='complex':
       x=random.random()
       M1=np.random.normal(0, np.cos(x), size=(alpha, beta))
       M2=1j*np.random.normal(0, np.sin(x), size=(alpha, beta))
       M=M1+M2
       return np.dot(M,M.conj().T)

def data(alpha,beta,normal):
    rho=W(alpha,beta,normal)/beta
    Eig=np.linalg.eigvals(rho)
    Eig=np.real(Eig)
    Eig=np.sort(Eig)
    data=list(Eig)
    return data

data3=data(10000,33333,normal)
data6=data(10000,16666,normal)
data10=data(10000,10000,normal)
data13=data(10000,5000,normal)

fig, axs = plt.subplots(2, 2)
axs[0, 0].hist(data3,density=True,bins=np.arange(data3[0], data3[-1], (data3[-1]-data3[0])/500),align='mid',color='skyblue')
x=np.linspace((1-np.sqrt(0.3))**2,(1+np.sqrt(0.3))**2,1000)
axs[0, 0].plot(x,np.sqrt(((1+np.sqrt(0.3))**2-x)*(x-(1-np.sqrt(0.3))**2))/(0.3*2*np.pi*x),'-',color='blueviolet')
axs[0, 0].set_ylim(0,1)
axs[0, 0].set_title(''r'$\lambda=0.3$')

axs[0, 1].hist(data6,density=True,bins=np.arange(data6[0], data6[-1], (data6[-1]-data6[0])/500),align='mid',color='skyblue')
x=np.linspace((1-np.sqrt(0.6))**2,(1+np.sqrt(0.6))**2,1000)
axs[0, 1].plot(x,np.sqrt(((1+np.sqrt(0.6))**2-x)*(x-(1-np.sqrt(0.6))**2))/(0.6*2*np.pi*x),'-',color='blueviolet')
axs[0, 1].set_ylim(0,1.25)
axs[0, 1].set_title(''r'$\lambda=0.6$')

axs[1, 0].hist(data10,density=True,bins=np.arange(data10[0], data10[-1], (data10[-1]-data10[0])/500),align='mid',color='skyblue')
x=np.linspace(0.001,(2)**2,1000)
axs[1, 0].plot(x,np.sqrt((2**2-x)*x)/(2*np.pi*x),'-',color='blueviolet')
axs[1, 0].set_ylim(0,2.5)
axs[1, 0].set_title(''r'$\lambda=1.0$')

axs[1, 1].hist(data13,density=True,bins=np.arange(data13[0], data13[-1], (data13[-1]-data13[0])/500),align='mid',color='skyblue')
x=np.linspace((1-np.sqrt(2))**2,(1+np.sqrt(2))**2,1000)
axs[1, 1].plot(x,np.sqrt(((1+np.sqrt(2))**2-x)*(x-(1-np.sqrt(2))**2))/(2*2*np.pi*x),'-',color='blueviolet')
axs[1, 1].set_ylim(0,1.25)
axs[1, 1].set_title(''r'$\lambda=2.0$')

plt.tight_layout() 
plt.savefig('FIG_1.pdf')

print("--- %s seconds ---" % (time.time() - start_time))
