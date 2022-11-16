import sympy
import numpy as np
from qibo import matrices, hamiltonians, symbols
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from statistics import mean
from numpy.linalg import matrix_rank
import csv

start_time = time.time()

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
    vals,vecs=tf.linalg.eigh(hs)
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
    M=tf.linalg.matmul(M, M.conj().T)
    M=M/np.trace(M)
    return M

def spectrum(rho):
    w=tf.linalg.eigvalsh(rho)
    w=tf.math.real(w)
    vp=np.array(w)
    data=list(vp)
    return data

def Svon(data):
    S=0
    for x in data:
        if x>0:
           S=S+x*np.log(x)
    return max(data),-S

def transition(n,i,p):
    control, solution, clauses=read_file(n, i)
    times_list=times(n, clauses)
    h0=h_initial(n, times_list)
    hp=h_problem(n, clauses)
    X=[]
    Y=[]
    for s in np.linspace(0, 1, num=11):
        HS=hs(s,h0,hp)
        val,vec=ground_state(HS)
        state=partition(vec,n,p)
        RHO=rho(state,n)
        data=spectrum(RHO)
        x,y=Svon(data)
        X.append(x)
        Y.append(y)
    return X,Y


def f(a,l):
    return -1/2*l*(4*np.log(2)-1)-l*np.log(l)+(l-1)*np.log(-4*(l-1)/a)+2*np.log(2)-1/2
    
i=1
n=12
a=2**int(n/2)
P=all_partition_set(n)
p=P[0]
X,Y=transition(n,i,p)

l=np.linspace(0.02,0.999,1000)
plt.figure(1,figsize=(4.5,4.5))
plt.axhline(y=np.log(64)-0.5,color='gray',linestyle='dashed')
plt.plot(l,f(a,l),'-',color='skyblue',label=''r'$\beta=64$')
plt.plot(X,Y,'+',color='blueviolet',markersize=5,label='adiabatic')
plt.title(''r'$\alpha=64$')
plt.xlim(-0.05,1.05)
plt.xlabel(''r'$\lambda_0$')
plt.ylabel('entropy')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
plt.tight_layout()
plt.savefig('FIG_9.pdf')

print("--- %s seconds ---" % (time.time() - start_time))
