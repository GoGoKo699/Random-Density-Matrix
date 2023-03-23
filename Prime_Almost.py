import numpy as np
import math
import cmath
import random
import time
from statistics import mean
import matplotlib.pyplot as plt

start_time = time.time()

alpha=8192

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
    #a_QFT=np.fft.fft(a)
    #a_QFT=a_QFT/np.linalg.norm(a_QFT)
    return a #,a_QFT
    
def build_state(state,n):
    limit = 2**n
    norm = int(2**(n/2))
    pp = (0+0j)*np.ones((norm,norm))
    for i in np.nonzero(state)[0]:
        b = np.int64(i % norm)
        a = np.int64((i - b) / norm)
        pp[a, b] =state[i]
    return pp

def H(m,n):
    N=2**n
    Matrix=[0]*int(N-m)+[1/np.sqrt(m)]*m
    random.shuffle(Matrix)
    Matrix=np.array(Matrix)
    return Matrix

def rho(state,n):
    M=build_state(state,n)
    M=np.matmul(M, M.conj().T)
    #M = np.float32(M)
    return M

def Svon(rho):
    S=0
    vp=np.linalg.eigvals(rho)
    vp=np.sort(vp)
    for x in vp:
        if 1e-15> x.real > -1e-15: 
           S=S
        else: 
           S=S+(abs(x))*math.log(abs(x))
    if abs(vp[-2])==0:
       gap=2*np.log(alpha)
    else:
       gap=np.log(vp[-1])-np.log(abs(vp[-2]))
       if gap>2*np.log(alpha):
          gap=2*np.log(alpha)
    return vp[-1],gap,-S

def f(a,l):
    return -1/2*l*(4*np.log(2)-1)-l*np.log(l)+(l-1)*np.log(-4*(l-1)/a)+2*np.log(2)-1/2
    
def f_gap(alpha,beta,l):
    return np.log(l)-np.log(((1-l)/alpha)*(1+np.sqrt(alpha/beta))**2)
    
def f_gap_inf(alpha,l):
    return np.log(l)-np.log((1-l)/alpha)

n=26

LIST=classify(2**n)

lambda_prime=[]
gap_prime=[]
ent_prime=[]
lambda_prime_QFT=[]
gap_prime_QFT=[]
ent_prime_QFT=[]
for i in range(len(LIST)):
    List=LIST[-1-i]
    state,state_QFT=State(n,List)
    state=State(n,List)
    x,g,y=Svon(rho(state,n))
    lambda_prime.append(x)
    gap_prime.append(g)
    ent_prime.append(y)
    x_QFT,g_QFT,y_QFT=Svon(rho(state_QFT,n))
    lambda_prime_QFT.append(x_QFT)
    gap_prime_QFT.append(g_QFT)
    ent_prime_QFT.append(y_QFT)
    
lambda_sum=[]
gap_sum=[]
ent_sum=[]
lambda_sum_QFT=[]
gap_sum_QFT=[]
ent_sum_QFT=[]

List=[]
for i in range(len(LIST)):
    List=List+LIST[i]
    state,state_QFT=State(n,List)
    state=State(n,List)
    x,g,y=Svon(rho(state,n))
    lambda_sum.append(x)
    gap_sum.append(g)
    ent_sum.append(y)
    x_QFT,g_QFT,y_QFT=Svon(rho(state_QFT,n))
    lambda_sum_QFT.append(x_QFT)
    gap_sum_QFT.append(g_QFT)
    ent_sum_QFT.append(y_QFT)
    
x=np.linspace(0.000001,0.999999,1000)

fig=plt.figure(figsize=(4.5,5))
ax = fig.add_subplot()

plt.fill_between(x, f(alpha,x),color='skyblue',alpha=0.3)
plt.fill_between(x, -x*np.log(x)-(1-x)*np.log(1-x),color='silver')
plt.fill_between(x, -x*np.log(x)-(1-x)*np.log((1-x)/(alpha)),np.log(alpha),color='silver')
plt.fill_between(x, f(alpha,x),-x*np.log(x)-(1-x)*np.log((1-x)/(alpha)),color='silver')
plt.fill_between(x, -np.log(x),color='silver')
plt.fill_between(x, 0, np.log(alpha), where=(x<4/alpha), color='silver')

plt.axvline(x=4/alpha,color='gray',linestyle='dotted') 
plt.plot(x,-x*np.log(x)-(1-x)*np.log(1-x),color='gray',linestyle='dashdot')
plt.plot(x,-np.log(x),color='gray',linestyle='dashed')
plt.plot(x,-x*np.log(x)-(1-x)*np.log((1-x)/(alpha)),color='gray',linestyle='solid')

x=np.linspace(4/alpha,0.999999,1000)
plt.plot(x,f(alpha,x),'-',color='skyblue',linewidth=1,label=''r'$\beta=8192$')

plt.plot(lambda_prime_QFT,ent_prime_QFT,'x-',markersize=7,color='gray')
plt.plot(lambda_sum_QFT,ent_sum_QFT,'*-',markersize=7,color='gray')
plt.plot(lambda_sum_QFT[0],ent_sum_QFT[0],'+',markersize=9,color='gray')

plt.plot(lambda_prime,ent_prime,'x-',markersize=5,color='forestgreen',label=''r'$\mathbb{P}_k$')
plt.plot(lambda_sum,ent_sum,'*-',markersize=5,color='blueviolet',label=''r'$\mathbb{U}_k$')
plt.plot(lambda_sum[0],ent_sum[0],'+',markersize=7,color='k')

plt.xlim(0,1)
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0])
plt.ylim(0,np.log(alpha))
plt.xlabel(''r'$\lambda_0$')
plt.ylabel('entropy')
plt.title(''r'$\alpha=8192$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fancybox=True, shadow=True, ncol=5)
ax.set_box_aspect(1)
plt.tight_layout()
#FIG. 16
plt.savefig('Prime_Almost.pdf')

fig=plt.figure(2,figsize=(4.5,5))
ax = fig.add_subplot()
x=np.linspace(0.000001,0.999999,1000)
plt.fill_between(x, f_gap(alpha,alpha,x),color='skyblue',alpha=0.3)
plt.fill_between(x, np.log(x)-np.log(1-x),color='silver')
plt.fill_between(x, f_gap(alpha,alpha,x),2*np.log(alpha),color='silver')
plt.plot(x,np.log(x)-np.log(1-x),color='gray',linestyle='dashdot')
plt.plot(x,np.log(x)-np.log((1-x)/alpha),color='gray',linestyle='solid')
plt.axvline(x=4/alpha,color='gray',linestyle='dotted')
x=np.linspace(4/alpha,0.999999,1000)
plt.plot(x,f_gap(alpha,alpha,x),'-',color='skyblue',linewidth=1,label=''r'$\beta=8192$')

plt.plot(lambda_prime,gap_prime,'x-',markersize=5,color='forestgreen',label=''r'$\mathbb{P}_k$')
plt.plot(lambda_sum,gap_sum,'*-',markersize=5,color='blueviolet',label=''r'$\mathbb{U}_k$')
plt.plot(lambda_sum[0],gap_sum[0],'+',markersize=7,color='k')

plt.xlim(0,1)
plt.ylim(0,2*np.log(alpha))
plt.yticks([0,4,8,12,16])
plt.xlabel(''r'$\lambda_0$')
plt.ylabel('gap')
plt.title(''r'$\alpha=8192$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fancybox=True, shadow=True, ncol=5)
ax.set_box_aspect(1)
plt.tight_layout()
#FIG. 20(d)
plt.savefig('Prime_Almost_gap.pdf')

print("--- %s seconds ---" % (time.time() - start_time))
