import numpy as np
import random
import time
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank

start_time = time.time()

def Primestate(N):
    a=np.ones((N, 1))
    a[0]=0
    a[1]=0
    b=int(np.sqrt(N))+1
    for i in range(2,b):
        c=int(N/i)+1
        for j in range(i,c):
             d=i*j
             if d<N: a[d]=0
    return a,np.count_nonzero(a)

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
    print(matrix_rank(M.T))
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
    return abs(vp[-1]),-S

def distribution(state,n,L):
    X=[]
    Y=[]
    for l in L:
        new_state=partition(state,n,l)
        x,y=Svon(rho(new_state,n))
        X.append(x)
        Y.append(y)
    return X,Y

def f(a,l):
    return -1/2*l*(4*np.log(2)-1)-l*np.log(l)+(l-1)*np.log(-4*(l-1)/a)+2*np.log(2)-1/2

n=14
L=all_partition_set(n)
prime_state,Pi=Primestate(2**n)
random_state=Randomstate(n,Pi)
X_p,Y_p=distribution(prime_state,n,L)

l=np.linspace(0.02,0.999,1000)
plt.figure(2,figsize=(4.5,4.5))

plt.axhline(y=np.log(128)-0.5,color='gray',linestyle='dashed')
plt.plot(l,f(128,l),'-',color='skyblue',linewidth=1,label=''r'$\alpha$')
plt.plot(l,f(65,l),'-',color='blueviolet',linewidth=1,label=''r'$\alpha_{rank}=65$')
plt.plot(X_p,Y_p,'+',color='blueviolet',markersize=5,label=''r'$| \mathbb{P}_n >$')
plt.xlim(-0.05,1.05)
plt.xlabel(''r'$\lambda_0$')
plt.ylabel('entropy')
plt.title(''r'$\alpha=\beta=128$')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
plt.tight_layout()
plt.savefig('FIG_10.pdf')

print("--- %s seconds ---" % (time.time() - start_time))
