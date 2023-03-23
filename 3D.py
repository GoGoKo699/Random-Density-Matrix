import numpy as np
import math
import cmath
import random
import time
import matplotlib.pyplot as plt

start_time = time.time()

normal='real'

def classify(N):
    List=np.ones(N)
    List=List.astype(int)
    List[0]=0
    List[1]=0
    b=int(math.sqrt(N))+1
    for i in range(2,b):
        c=int(N/i)+1
        for j in range(i,c):
             d=i*j
             if d<N: List[d]=0
    for k in range(2,1+int(math.log2(N))):
        for i in np.where(List==1)[0]:
            x=List[i:int(N/i)+1]
            l=np.where(x==k-1)[0]
            for j in l:
                d=i*(j+i)
                if d<N: List[d]=k
    LIST=[]
    for k in range(1,int(math.log2(N))):
        LIST.append(list(np.where(List==k)[0]))
    return LIST
    
def State(n,List):
    a=np.zeros(2**n)
    m=len(List)
    for i in List:
        a[i]=1/math.sqrt(m)
    return a
    
def build_state(state,n):
    limit = 2**n
    norm = int(2**(n/2))
    pp = np.zeros((norm,norm))
    for i in np.nonzero(state)[0]:
        b = np.int64(i % norm)
        a = np.int64((i - b) / norm)
        pp[a, b] =state[i]
    return pp

def rho(state,n):
    M=build_state(state,n)
    M=np.matmul(M, M.T)
    return M
    
def Svon(rho):
    vp=np.linalg.eigvals(rho)
    vp=np.real(vp)
    vp=np.sort(vp)
    return list(vp)
    
def RM(alpha,l0,normal):
    if normal=='real':
       sigma=np.sqrt((1-l0)/alpha)
       gamma=random.choice([-1,1])*np.sqrt(l0/alpha)
       M=np.random.normal(gamma, sigma, size=(alpha,alpha))
       rho=np.dot(M,M.T)
       rho=rho/np.trace(rho)
       return Svon(rho)
    if normal=='complex':
       sigma=np.sqrt((1-l0)/alpha)
       gamma=random.choice([-1,1])*np.sqrt(l0/alpha)
       M1=np.random.normal(gamma, sigma, size=(alpha,alpha))
       M2=1j*np.random.normal(gamma, sigma, size=(alpha,alpha))
       M=(M1+M2)/2
       rho=np.dot(M,M.conj().T)
       rho=rho/np.trace(rho)
       return Svon(rho)
       
def LB(l):
    SUM=l
    vp=[l]
    while SUM!=1:
          t=min(l,1-SUM)
          SUM=SUM+t
          vp.append(t)
    return vp
    
def Renyi(eig,d):
    SUM=0
    for p in eig:
        if p>1e-10:
           SUM=SUM+p**d
    SUM=np.log(SUM)
    SUM=SUM/(1-d)
    return max(eig),SUM
    
n=14
alpha=int(2**(n/2))
D1=np.linspace(0.001, 0.999, 1001)
D2=np.linspace(1.001, np.log(128), 1001)
x=np.linspace(4/alpha, 1, 1000)

fig=plt.figure(1,figsize=(7,3))
ax1 = fig.add_subplot(121,projection='3d')
ax2 = fig.add_subplot(122,projection='3d')

LIST=classify(2**n)

EIG_P_K=[]
for i in range(len(LIST)):
    List=LIST[-1-i]
    state=State(n,List)
    eig=Svon(rho(state,n))
    EIG_P_K.append(eig)
    
EIG_P_U=[]
List=[]
for i in range(len(LIST)):
    List=List+LIST[i]
    state=State(n,List)
    eig=Svon(rho(state,n))
    EIG_P_U.append(eig)

EIG_MPD=[]
for i in range(len(x)):
    eig=RM(alpha,x[i],normal)
    EIG_MPD.append(eig)
    
EIG_L_B=[]
for i in range(len(x)):
    eig=LB(x[i])
    EIG_L_B.append(eig)
    
#plot 1
    
for i in range(len(LIST)):
    X_P_U=[]
    Y_P_U=[]
    for d in D1:
        x,y=Renyi(EIG_P_U[i],d)
        X_P_U.append(x)
        Y_P_U.append(y)
    Z_P_U=D1
    ax1.plot(X_P_U, Y_P_U, Z_P_U, '-',color='blueviolet',linewidth=1,alpha=0.5)
    X_P_K=[]
    Y_P_K=[]
    for d in D1:
        x,y=Renyi(EIG_P_K[i],d)
        X_P_K.append(x)
        Y_P_K.append(y)
    Z_P_K=D1
    ax1.plot(X_P_K, Y_P_K, Z_P_K, '-',color='forestgreen',linewidth=1,alpha=0.5)

for d in D1:
    X_MPD=[]
    Y_MPD=[]
    for i in range(len(EIG_MPD)):
        x,y=Renyi(EIG_MPD[i],d)
        X_MPD.append(x)
        Y_MPD.append(y)
    Z_MPD=[d]*len(X_MPD)
    ax1.plot(X_MPD, Y_MPD, Z_MPD, '-',color='skyblue',linewidth=1,alpha=0.025)
    X_L_B=[]
    Y_L_B=[]
    for i in range(len(EIG_L_B)):
        x,y=Renyi(EIG_L_B[i],d)
        X_L_B.append(x)
        Y_L_B.append(y)
    Z_L_B=[d]*len(X_L_B)
    ax1.plot(X_L_B, Y_L_B, Z_L_B, '-',color='gray',linewidth=1,alpha=0.02)

ax1.set_xlim(0, 1)
ax1.set_ylim(0, np.log(alpha))
ax1.set_zlim(0, 1)
ax1.set_zticks([0,1])
ax1.set_xlabel(''r'$\lambda_0$')
ax1.set_ylabel('entropy')
ax1.set_zlabel('d')

#plot 2
    
for i in range(len(LIST)):
    X_P_U=[]
    Y_P_U=[]
    for d in D2:
        x,y=Renyi(EIG_P_U[i],d)
        X_P_U.append(x)
        Y_P_U.append(y)
    Z_P_U=D2
    ax2.plot(X_P_U, Y_P_U, Z_P_U, '-',color='blueviolet',linewidth=1,alpha=0.5)
    X_P_K=[]
    Y_P_K=[]
    for d in D2:
        x,y=Renyi(EIG_P_K[i],d)
        X_P_K.append(x)
        Y_P_K.append(y)
    Z_P_K=D2
    ax2.plot(X_P_K, Y_P_K, Z_P_K, '-',color='forestgreen',linewidth=1,alpha=0.5)

for d in D2:
    X_MPD=[]
    Y_MPD=[]
    for i in range(len(EIG_MPD)):
        x,y=Renyi(EIG_MPD[i],d)
        X_MPD.append(x)
        Y_MPD.append(y)
    Z_MPD=[d]*len(X_MPD)
    ax2.plot(X_MPD, Y_MPD, Z_MPD, '-',color='skyblue',linewidth=1,alpha=0.025)
    X_L_B=[]
    Y_L_B=[]
    for i in range(len(EIG_L_B)):
        x,y=Renyi(EIG_L_B[i],d)
        X_L_B.append(x)
        Y_L_B.append(y)
    Z_L_B=[d]*len(X_L_B)
    ax2.plot(X_L_B, Y_L_B, Z_L_B, '-',color='gray',linewidth=1,alpha=0.02)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, np.log(alpha))
ax2.set_zlim(1, np.log(128))
ax2.set_zticks([1,2,3,4])
ax2.set_xlabel(''r'$\lambda_0$')
ax2.set_ylabel('entropy')
ax2.set_zlabel('d')

plt.tight_layout()
#FIG. 22
plt.savefig('3D.pdf')

print("--- %s seconds ---" % (time.time() - start_time))
