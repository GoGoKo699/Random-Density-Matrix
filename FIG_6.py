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
       M=random.choice([-1,1])*np.random.normal(gamma, sigma, alpha*beta)
       state=M/np.linalg.norm(M)
       return state
    if normal=='complex':
       l0=random.uniform(0,1)
       sigma=np.sqrt((1-l0)/alpha)
       gamma=random.choice([-1,1])*np.sqrt(l0/alpha)
       M1=np.random.normal(gamma, sigma, alpha*beta)
       M2=1j*np.random.normal(gamma, sigma, alpha*beta)
       M=(M1+M2)/2
       state=M/np.linalg.norm(M)
       return state

def Svon(state,alpha,beta):
    state=state.reshape((alpha,beta))
    rho=np.dot(state,state.conj().T)
    rho=rho/np.trace(rho)
    Eig=np.linalg.eigvals(rho)
    Eig=np.real(Eig)
    Eig=np.sort(Eig)
    S=0
    for x in Eig:
        if 1e-15> x.real > -1e-15: 
           S=S
        else: 
           S=S+(abs(x))*np.log(abs(x))
    return Eig[-1],-S

def f(a,b,l):
    result=4*a*b*l*np.log(l)+ (8*a*b*np.log(2) - a**2 - b**2)*(l-1)
    result=result-4*(a*b*l - a*b)*np.log(-2*((a + b)*l - (b-a)*(1-l) - a - b)/(a*b))
    result=result- (a+b)*(b-a)*(1-l)
    return -1/4*result/(a*b)

def f_inf(a,l):
    result=(-2*np.log(2)-np.log(a)-np.log(l))*l+(l-1)*np.log(4-4*l)+2*np.log(2)+np.log(a)
    return result

alpha=128

beta=128
x_128=[]
y_128=[]
x_128_QFT=[]
y_128_QFT=[]
for i in range(10):
    state=RM(alpha,beta,normal)
    x,y=Svon(state,alpha,beta)
    x_128.append(x)
    y_128.append(y)
    state=np.fft.fft(state)
    state=state/np.linalg.norm(state)
    x,y=Svon(state,alpha,beta)
    x_128_QFT.append(x)
    y_128_QFT.append(y)

beta=256
x_256=[]
y_256=[]
x_256_QFT=[]
y_256_QFT=[]
for i in range(10):
    state=RM(alpha,beta,normal)
    x,y=Svon(state,alpha,beta)
    x_256.append(x)
    y_256.append(y)
    state=np.fft.fft(state)
    state=state/np.linalg.norm(state)
    x,y=Svon(state,alpha,beta)
    x_256_QFT.append(x)
    y_256_QFT.append(y)

beta=8192
x_8192=[]
y_8192=[]
x_8192_QFT=[]
y_8192_QFT=[]
for i in range(10):
    state=RM(alpha,beta,normal)
    x,y=Svon(state,alpha,beta)
    x_8192.append(x)
    y_8192.append(y)
    state=np.fft.fft(state)
    state=state/np.linalg.norm(state)
    x,y=Svon(state,alpha,beta)
    x_8192_QFT.append(x)
    y_8192_QFT.append(y)

l=np.linspace(0.02,0.999,1000)

plt.figure(1,figsize=(4.5,4.5))
plt.axhline(y=np.log(alpha)-0.5,color='gray',linestyle='dashed') 
plt.plot(l,f(alpha,128,l),'-',color='skyblue',linewidth=1)
plt.plot(l,f(alpha,256,l),'-',color='forestgreen',linewidth=1)
plt.plot(l,f_inf(alpha,l),'-',color='blueviolet',linewidth=1)  
plt.plot(x_128,y_128,'+',color='skyblue',markersize=5,label=''r'$\beta=128$')
plt.plot(x_128_QFT,y_128_QFT,'k+',markersize=5)
plt.plot(x_256,y_256,'+',color='forestgreen',markersize=5,label=''r'$\beta=256$')
plt.plot(x_256_QFT,y_256_QFT,'k+',markersize=5)
plt.plot(x_8192,y_8192,'+',color='blueviolet',markersize=5,label=''r'$\beta\rightarrow\infty$')
plt.plot(x_8192_QFT,y_8192_QFT,'k+',markersize=5,label=''r'QFT')
plt.xlim(-0.05,1.05)
plt.xlabel(''r'$\lambda_0$')
plt.ylabel('entropy')
plt.title(''r'$\alpha=128$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
plt.tight_layout() 
plt.savefig('FIG_6.pdf')

print("--- %s seconds ---" % (time.time() - start_time))
