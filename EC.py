import sympy
import numpy as np
from qibo import matrices, hamiltonians, symbols
import matplotlib.pyplot as plt
import time
import random

start_time = time.time()

alpha=64

def read_file(file_name, instance):
    file = open('{q}bit/n{q}i{i}.txt'.format(q=file_name, i=instance), 'r')
    control = list(map(int, file.readline().split()))
    solution = list(map(str, file.readline().split()))
    clauses = [list(map(int, file.readline().split())) for _ in range(control[1])]
    return control, solution, clauses

def times(n, clauses):
    times = np.zeros(n)
    for clause in clauses:
        for num in clause:
            times[num-1] += 1
    return times

def h_initial(n, times_list):
    symbolic_h0=sum(0.5 * times_list[i] * (1 - symbols.X(i)) for i in range(n))
    h0 = hamiltonians.SymbolicHamiltonian(symbolic_h0)
    return h0.matrix

def h_grover(n):
    h1 = np.identity(2**n)
    h1[0][0]=0
    return h1

def h_problem(n, clauses):
    z_matrix = (matrices.I - matrices.Z) / 2.0
    z = [symbols.Symbol(i, z_matrix) for i in range(n)]
    symbolic_hp=sum((sum(z[i - 1] for i in clause) - 1) ** 2 for clause in clauses)
    hp = hamiltonians.SymbolicHamiltonian(symbolic_hp)
    return hp.matrix

def hs(s,h0,hp):
    return (1-s)*h0+s*hp

def ground_state(hs):
    vals,vecs=np.linalg.eigh(hs)
    val=float(list(vals)[0])
    vec=np.array(vecs[:,0])
    return val,vec

def all_partition_set(n):
    a=int(n/2)
    L=[]
    for i in range(2**n):
        m=[int(x) for x in str(bin(i))[2:]]
        m=[0]*(n-len(m))+m
        if m.count(0)==a:
           L.append(m)
    return L

def partition(state,n,p):
    N=2**n
    c=[0]*N
    for i in range(2**n):
        a=[]
        m=[int(x) for x in str(bin(i))[2:]]
        m=[0]*(n-len(m))+m
        for j in range(n):
            if p[j]==1: a.append(m[j])
        for j in range(n):
            if p[j]==0: a.append(m[j])
        x=int(''.join(map(str, a)), 2)
        c[x]=state[i]
    return np.array(c)

def rho(state,n):
    M=state.reshape((int(2**(n/2)), int(2**(n/2))))
    M=M @ M.conj().T
    M=M/np.trace(M)
    return M

def spectrum(rho):
    w=np.linalg.eigvalsh(rho)
    vp=np.array(w)
    vp=np.sort(vp)
    data=list(vp)
    return data

def Svon(vp):
    S=0
    for x in vp:
        if x>0:
           S=S+x*np.log(x)
    if abs(vp[-2])==0:
       gap=2*np.log(alpha)
    else:
       gap=np.log(vp[-1])-np.log(abs(vp[-2]))
       if gap>2*np.log(alpha):
          gap=2*np.log(alpha)
    return vp[-1],gap,-S

def transition(n,i,p):
    control, solution, clauses=read_file(n, i)
    times_list=times(n, clauses)
    h0=h_initial(n, times_list)
    hp=h_problem(n, clauses)
    p_0=p[0]
    p_1=p[1]
    p_2=p[2]
    X_0=[]
    G_0=[]
    Y_0=[]
    X_1=[]
    G_1=[]
    Y_1=[]
    X_2=[]
    G_2=[]
    Y_2=[]
    for s in np.linspace(0, 1, num=11):
        HS=hs(s,h0,hp)
        val,vec=ground_state(HS)
        state_0=partition(vec,n,p_0)
        RHO_0=rho(state_0,n)
        data=spectrum(RHO_0)
        x,g,y=Svon(data)
        X_0.append(x)
        G_0.append(g)
        Y_0.append(y)
        state_1=partition(vec,n,p_1)
        RHO_1=rho(state_1,n)
        data=spectrum(RHO_1)
        x,g,y=Svon(data)
        X_1.append(x)
        G_1.append(g)
        Y_1.append(y)
        state_2=partition(vec,n,p_2)
        RHO_2=rho(state_2,n)
        data=spectrum(RHO_2)
        x,g,y=Svon(data)
        X_2.append(x)
        G_2.append(g)
        Y_2.append(y)
    return X_0,G_0,Y_0,X_1,G_1,Y_1,X_2,G_2,Y_2


def f(a,l):
    return -1/2*l*(4*np.log(2)-1)-l*np.log(l)+(l-1)*np.log(-4*(l-1)/a)+2*np.log(2)-1/2
    
def f_gap(a,b,l):
    return np.log(l)-np.log(((1-l)/a)*(1+np.sqrt(a/b))**2)
    
def f_gap_inf(a,l):
    return np.log(l)-np.log((1-l)/a)

P=all_partition_set(12)
p=random.sample(P,3)

X_1_0,G_1_0,Y_1_0,X_1_1,G_1_1,Y_1_1,X_1_2,G_1_2,Y_1_2=transition(12,1,p)
X_2_0,G_2_0,Y_2_0,X_2_1,G_2_1,Y_2_1,X_2_2,G_2_2,Y_2_2=transition(12,2,p)
X_3_0,G_3_0,Y_3_0,X_3_1,G_3_1,Y_3_1,X_3_2,G_3_2,Y_3_2=transition(12,3,p)

x=np.linspace(0.000001,0.999999,1000)
fig=plt.figure(1,figsize=(4.5,5))
ax = fig.add_subplot()

plt.fill_between(x, -x*np.log(x)-(1-x)*np.log(1-x),color='silver')
plt.fill_between(x, -np.log(x),color='silver')
plt.fill_between(x, -x*np.log(x)-(1-x)*np.log((1-x)/alpha),np.log(alpha),color='silver')
plt.plot(x,-x*np.log(x)-(1-x)*np.log(1-x),color='gray',linestyle='dashdot')
plt.plot(x,-np.log(x),color='gray',linestyle='dashed')
plt.plot(x,(1-x)*np.log(alpha)-x*np.log(x)-(1-x)*np.log(1-x),color='gray',linestyle='solid')

plt.plot(X_1_0,Y_1_0,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_1_0,Y_1_0,'+',color='forestgreen',markersize=5,label='i=1')
plt.plot(X_1_1,Y_1_1,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_1_1,Y_1_1,'+',color='forestgreen',markersize=5)
plt.plot(X_1_2,Y_1_2,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_1_2,Y_1_2,'+',color='forestgreen',markersize=5)

plt.plot(X_2_0,Y_2_0,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_2_0,Y_2_0,'x',color='forestgreen',markersize=5,label='i=2')
plt.plot(X_2_1,Y_2_1,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_2_1,Y_2_1,'x',color='forestgreen',markersize=5)
plt.plot(X_2_2,Y_2_2,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_2_2,Y_2_2,'x',color='forestgreen',markersize=5)

plt.plot(X_3_0,Y_3_0,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_3_0,Y_3_0,'*',color='forestgreen',markersize=5,label='i=3')
plt.plot(X_3_1,Y_3_1,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_3_1,Y_3_1,'*',color='forestgreen',markersize=5)
plt.plot(X_3_2,Y_3_2,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_3_2,Y_3_2,'*',color='forestgreen',markersize=5)

plt.title(''r'$\alpha=64$')
plt.xlim(0.7,1)
plt.ylim(0,1)
plt.xticks([0.7,0.8,0.9,1.0])
plt.yticks([0,1])
plt.xlabel(''r'$\lambda_0$')
plt.ylabel('entropy')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fancybox=True, shadow=True, ncol=5)
ax.set_box_aspect(1)
plt.tight_layout()
#FIG. 2
plt.savefig('EC_bis.pdf')

fig=plt.figure(2,figsize=(4.5,5))
ax = fig.add_subplot()

plt.fill_between(x, f(alpha,x),color='skyblue',alpha=0.3)
plt.fill_between(x, -x*np.log(x)-(1-x)*np.log(1-x),color='silver')
plt.fill_between(x, -np.log(x),color='silver')
plt.fill_between(x, f(alpha,x),np.log(alpha),color='silver')
plt.fill_between(x, 0, np.log(alpha), where=(x<4/alpha), color='silver')

plt.axvline(x=4/alpha,color='gray',linestyle='dotted')
plt.plot(x,-x*np.log(x)-(1-x)*np.log(1-x),color='gray',linestyle='dashdot')
plt.plot(x,-np.log(x),color='gray',linestyle='dashed')
plt.plot(x,(1-x)*np.log(alpha)-x*np.log(x)-(1-x)*np.log(1-x),color='gray',linestyle='solid')
x=np.linspace(4/64,0.999999,1000)
plt.plot(x,f(alpha,x),'-',color='skyblue',linewidth=1,label=''r'$\beta=64$')

plt.plot(X_1_0,Y_1_0,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_1_0,Y_1_0,'+',color='forestgreen',markersize=5,label='i=1')
plt.plot(X_1_1,Y_1_1,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_1_1,Y_1_1,'+',color='forestgreen',markersize=5)
plt.plot(X_1_2,Y_1_2,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_1_2,Y_1_2,'+',color='forestgreen',markersize=5)

plt.plot(X_2_0,Y_2_0,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_2_0,Y_2_0,'x',color='forestgreen',markersize=5,label='i=2')
plt.plot(X_2_1,Y_2_1,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_2_1,Y_2_1,'x',color='forestgreen',markersize=5)
plt.plot(X_2_2,Y_2_2,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_2_2,Y_2_2,'x',color='forestgreen',markersize=5)

plt.plot(X_3_0,Y_3_0,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_3_0,Y_3_0,'*',color='forestgreen',markersize=5,label='i=3')
plt.plot(X_3_1,Y_3_1,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_3_1,Y_3_1,'*',color='forestgreen',markersize=5)
plt.plot(X_3_2,Y_3_2,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_3_2,Y_3_2,'*',color='forestgreen',markersize=5)

plt.title(''r'$\alpha=64$')
plt.xlim(0,1)
plt.ylim(0,np.log(64))
plt.yticks([0,1,2,3])
plt.xlabel(''r'$\lambda_0$')
plt.ylabel('entropy')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fancybox=True, shadow=True, ncol=5)
ax.set_box_aspect(1)
plt.tight_layout()
#FIG. 13
plt.savefig('EC.pdf')

fig=plt.figure(3,figsize=(4.5*0.98,5))
ax = fig.add_subplot()
x=np.linspace(0.000001,0.999999,1000)
plt.fill_between(x, f_gap(alpha,alpha,x),color='skyblue',alpha=0.3)
plt.fill_between(x, np.log(x)-np.log(1-x),color='silver')
plt.fill_between(x, f_gap(alpha,alpha,x),2*np.log(alpha),color='silver')
plt.plot(x,np.log(x)-np.log(1-x),color='gray',linestyle='dashdot')
plt.plot(x,np.log(x)-np.log((1-x)/alpha),color='gray',linestyle='solid')
plt.axvline(x=4/alpha,color='gray',linestyle='dotted')
x=np.linspace(4/alpha,0.999999,1000)
plt.plot(x,f_gap(alpha,alpha,x),'-',color='skyblue',linewidth=1,label=''r'$\beta=64$')

plt.plot(X_1_0,G_1_0,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_1_0,G_1_0,'+',color='forestgreen',markersize=5,label='i=1')
plt.plot(X_1_1,G_1_1,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_1_1,G_1_1,'+',color='forestgreen',markersize=5)
plt.plot(X_1_2,G_1_2,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_1_2,G_1_2,'+',color='forestgreen',markersize=5)

plt.plot(X_2_0,G_2_0,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_2_0,G_2_0,'x',color='forestgreen',markersize=5,label='i=2')
plt.plot(X_2_1,G_2_1,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_2_1,G_2_1,'x',color='forestgreen',markersize=5)
plt.plot(X_2_2,G_2_2,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_2_2,G_2_2,'x',color='forestgreen',markersize=5)

plt.plot(X_3_0,G_3_0,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_3_0,G_3_0,'*',color='forestgreen',markersize=5,label='i=3')
plt.plot(X_3_1,G_3_1,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_3_1,G_3_1,'*',color='forestgreen',markersize=5)
plt.plot(X_3_2,G_3_2,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(X_3_2,G_3_2,'*',color='forestgreen',markersize=5)

plt.xlim(0,1)
plt.ylim(0,2*np.log(alpha))
plt.yticks([0,2,4,6,8])
plt.xlabel(''r'$\lambda_0$')
plt.ylabel('gap')
plt.title(''r'$\alpha=64$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fancybox=True, shadow=True, ncol=5)
ax.set_box_aspect(1)
plt.tight_layout()
#FIG. 20(a)
plt.savefig('EC_gap.pdf')

print("--- %s seconds ---" % (time.time() - start_time))
