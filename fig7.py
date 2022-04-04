import numpy as np
import random
import time
import matplotlib.pyplot as plt

def normal(alpha):
    M=np.random.normal(0, 1, size=(alpha, alpha))
    return (M+M.T)/(2*np.sqrt(alpha))

def normal2(alpha):
    M=np.random.normal(0, 2, size=(alpha, alpha))
    return (M+M.T)/(2*np.sqrt(alpha))

def complex_normal(alpha):
    x=random.random()
    M1=np.random.normal(0, np.cos(x), size=(alpha, alpha))
    M2=1j*np.random.normal(0, np.sin(x), size=(alpha, alpha))
    M=M1+M2
    return (M+M.conj().T)/(2*np.sqrt(alpha))

def unifrom(alpha):
    M=np.random.uniform(-np.sqrt(3), np.sqrt(3), size=(alpha, alpha))
    return (M+M.T)/(2*np.sqrt(alpha))

def data(rho):
    Eig=np.linalg.eigvals(rho)
    Eig=np.real(Eig)
    Eig=np.sort(Eig)
    data=list(Eig)
    return data

start_time = time.time()

alpha=10000

rho_n=normal(alpha)
data_n=data(rho_n)

rho_n2=normal2(alpha)
data_n2=data(rho_n2)

rho_cn=complex_normal(alpha)
data_cn=data(rho_cn)

rho_u=unifrom(alpha)
data_u=data(rho_n)

x=np.linspace(-np.sqrt(2),np.sqrt(2),1000)

fig, axs = plt.subplots(2, 2)
axs[0, 0].hist(data_n,density=True,bins=np.arange(data_n[0], data_n[-1], (data_n[-1]-data_n[0])/500),align='mid',color='b')
axs[0, 0].plot(x,np.sqrt(2-x**2)/np.pi,'m-')
axs[0, 0].set_title(''r'$\mathcal{N}(0,1)$')

axs[0, 1].hist(data_n2,density=True,bins=np.arange(data_n2[0], data_n2[-1], (data_n2[-1]-data_n2[0])/500),align='mid',color='b')
axs[0, 1].plot(x,np.sqrt(2-x**2)/np.pi,'m-')
y=np.linspace(-2*np.sqrt(2),2*np.sqrt(2),1000)
axs[0, 1].plot(y,np.sqrt(8-y**2)/(4*np.pi),'g-')
axs[0, 1].set_title(''r'$\mathcal{N}(0,2)$')

axs[1, 0].hist(data_cn,density=True,bins=np.arange(data_cn[0], data_cn[-1], (data_cn[-1]-data_cn[0])/500),align='mid',color='b')
axs[1, 0].plot(x,np.sqrt(2-x**2)/np.pi,'m-')
axs[1, 0].set_title(''r'$\mathcal{N}_{\mathbb{C}}(0,1)$')

axs[1, 1].hist(data_u,density=True,bins=np.arange(data_u[0], data_u[-1], (data_u[-1]-data_u[0])/500),align='mid',color='b')
axs[1, 1].plot(x,np.sqrt(2-x**2)/np.pi,'m-')
axs[1, 1].set_title(''r'$\mathcal{U}(-\sqrt{3},\sqrt{3})$')

plt.tight_layout() 
plt.savefig('FIG_7.pdf')

print("--- %s seconds ---" % (time.time() - start_time))
