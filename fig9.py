import numpy as np
import random
import time
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank

start_time = time.time()

def Primestate(N):
    a=np.ones((N, 1))
    a[0][0]=0
    a[1][0]=0
    b=int(np.sqrt(N))+1
    for i in range(2,b):
        c=int(N/i)+1
        for j in range(i,c):
             d=i*j
             if d<N: a[d][0]=0
    return a

def all_partition_set(n):
    a=int(n/2)
    L=[]
    for i in range(2**n):
        m=[int(x) for x in str(bin(i))[2:]]
        m=[0]*(n-len(m))+m
        if m.count(0)==a:
           L.append(m)
    return L

def partition(state,n,l):
    N=2**n
    c=[0]*N
    for i in np.nonzero(state)[0]:
        a=[]
        m=[int(x) for x in str(bin(i))[2:]]
        m=[0]*(n-len(m))+m
        for j in range(n):
            if l[j]==1: a.append(m[j])
        for j in range(n):
            if l[j]==0: a.append(m[j])
        x=int(''.join(map(str, a)), 2)
        c[x]=1
    return c

def build_state(state,n):
    limit = 2**n
    norm = int(2**(n/2))
    pp = np.zeros((norm, norm))
    for i in np.nonzero(state):
        b = np.int64(i % norm)
        a = np.int64((i - b) / norm)
        pp[a, b] = 1
    return pp

def rho(state,n):
    M=build_state(state,n)
    M=np.dot(M,M.T)
    M=M/np.trace(M)
    return M

def Svon(rho):
    S=0
    vp=np.linalg.eigvals(rho)
    vp=np.sort(vp)
    for x in vp:
        if 1e-15> x.real > -1e-15: 
           S=S
        else: 
           S=S+(abs(x))*np.log(abs(x))
    r=matrix_rank(rho)
    return abs(vp[-1]),-S,r

def distribution(state,n,L):
    X=[]
    Y=[]
    R=[]
    for l in L:
        new_state=partition(state,n,l)
        x,y,r=Svon(rho(new_state,n))
        X.append(x)
        Y.append(y)
        R.append(r)
    return X,Y,R

def f(a,l):
    return -1/2*l*(4*np.log(2)-1)-l*np.log(l)+(l-1)*np.log(-4*(l-1)/a)+2*np.log(2)-1/2

n=14
L=all_partition_set(n)
state=Primestate(2**n)
X,Y,R=distribution(state,n,L)
alpha=2**(n/2)/2+1

l=np.linspace(0.02,0.999,1000)
fig, (ax1, ax2) = plt.subplots(2,figsize=(6,8.25))

ax1.axhline(y=np.log(128)-0.5,color='gray',linestyle='dashed')
ax1.plot(l,f(2**(n/2),l),'-',color='skyblue',linewidth=1,label=''r'$\alpha$')
ax1.plot(l,f(alpha,l),'-',color='blueviolet',linewidth=1,label=''r'$\alpha_{rank}$')
ax1.plot(X,Y,'+',color='blueviolet',markersize=1,label=''r'$| \mathbb{P}_n >$')
ax1.set_xlim(-0.05,1.05)
ax1.set_xlabel(''r'$\lambda_0$')
ax1.set_ylabel('Shannon entropy')
ax1.set_title('Global')

ax2.plot(l,f(alpha,l),'-',color='blueviolet',linewidth=1)
ax2.plot(X,Y,'+',color='blueviolet',markersize=1)
ax2.set_xlim(0.24,0.255)
ax2.set_ylim(3.25,3.425)
ax2.set_xlabel(''r'$\lambda_0$')
ax2.set_ylabel('Shannon entropy')
ax2.set_title('Zoom')

ax1.legend(bbox_to_anchor=(1, 1))
plt.tight_layout() 
plt.savefig('FIG_6.pdf')

print("--- %s seconds ---" % (time.time() - start_time))
