import fractions
import Shor_functions
from qibo import callbacks
import numpy as np
import time
import math
import matplotlib.pyplot as plt

start_time = time.time()

n1_N1=133
n1_N2=143
n1_N3=221

n1_a1=16
n1_a2=58
n1_a3=101

n2_N1=299
n2_N2=319
n2_N3=493

n2_a1=37
n2_a2=96
n2_a3=200

n3_N1=611
n3_N2=713
n3_N3=899

n3_a1=127
n3_a2=256
n3_a3=400

def trajectory(N,a):
    print('N=',N)
    print('a=',a)
    n=int(np.ceil(np.log2(N)))
    entropy_shor=callbacks.EntanglementEntropy([n+1+i for i in range(n)], compute_spectrum=True)
    s=Shor_functions.quantum_order_finding_semiclassical(entropy_shor,N,a)
    sr=s/2**(2*n)
    r=fractions.Fraction.from_float(sr).limit_denominator(N).denominator
    print(f"Quantum circuit outputs r = {r}.\n")
    factors=Shor_functions.find_factors(r,a,N)
    entanglement=list(np.array(entropy_shor[:])*math.log(2))
    spectrum_shor=entropy_shor.spectrum
    entanglement_shor=list(np.array(entropy_shor[:])*math.log(2))
    spectrum_shor=entropy_shor.spectrum
    L0_shor=[]
    Gap_shor=[]
    for i in range(len(spectrum_shor)):
        List=list(np.sort(spectrum_shor[i]))
        if len(List)==1:
           Gap_shor.append(2*np.log(2**n))
        else:
           Gap_shor.append(List[1]-List[0])
        L0=math.exp(-List[0])
        L0_shor.append(L0)
    return L0_shor,Gap_shor,entanglement_shor
    
def f(a,b,l):
    result=(1-l)*(np.log(a)-np.log(1-l)-a/(2*b))-l*np.log(l)
    return result
    
def f_gap(a,b,l):
    return np.log(l)-np.log(((1-l)/a)*(1+np.sqrt(a/b))**2)
    
L_n1_N1_a1,gap_n1_N1_a1,entanglement_n1_N1_a1=trajectory(n1_N1,n1_a1)
L_n1_N1_a2,gap_n1_N1_a2,entanglement_n1_N1_a2=trajectory(n1_N1,n1_a2)
L_n1_N1_a3,gap_n1_N1_a3,entanglement_n1_N1_a3=trajectory(n1_N1,n1_a3)

L_n1_N2_a1,gap_n1_N2_a1,entanglement_n1_N2_a1=trajectory(n1_N2,n1_a1)
L_n1_N2_a2,gap_n1_N2_a2,entanglement_n1_N2_a2=trajectory(n1_N2,n1_a2)
L_n1_N2_a3,gap_n1_N2_a3,entanglement_n1_N2_a3=trajectory(n1_N2,n1_a3)

L_n1_N3_a1,gap_n1_N3_a1,entanglement_n1_N3_a1=trajectory(n1_N3,n1_a1)
L_n1_N3_a2,gap_n1_N3_a2,entanglement_n1_N3_a2=trajectory(n1_N3,n1_a2)
L_n1_N3_a3,gap_n1_N3_a3,entanglement_n1_N3_a3=trajectory(n1_N3,n1_a3)

L_n2_N1_a1,gap_n2_N1_a1,entanglement_n2_N1_a1=trajectory(n2_N1,n2_a1)
L_n2_N1_a2,gap_n2_N1_a2,entanglement_n2_N1_a2=trajectory(n2_N1,n2_a2)
L_n2_N1_a3,gap_n2_N1_a3,entanglement_n2_N1_a3=trajectory(n2_N1,n2_a3)

L_n2_N2_a1,gap_n2_N2_a1,entanglement_n2_N2_a1=trajectory(n2_N2,n2_a1)
L_n2_N2_a2,gap_n2_N2_a2,entanglement_n2_N2_a2=trajectory(n2_N2,n2_a2)
L_n2_N2_a3,gap_n2_N2_a3,entanglement_n2_N2_a3=trajectory(n2_N2,n2_a3)

L_n2_N3_a1,gap_n2_N3_a1,entanglement_n2_N3_a1=trajectory(n2_N3,n2_a1)
L_n2_N3_a2,gap_n2_N3_a2,entanglement_n2_N3_a2=trajectory(n2_N3,n2_a2)
L_n2_N3_a3,gap_n2_N3_a3,entanglement_n2_N3_a3=trajectory(n2_N3,n2_a3)

L_n3_N1_a1,gap_n3_N1_a1,entanglement_n3_N1_a1=trajectory(n3_N1,n3_a1)
L_n3_N1_a2,gap_n3_N1_a2,entanglement_n3_N1_a2=trajectory(n3_N1,n3_a2)
L_n3_N1_a3,gap_n3_N1_a3,entanglement_n3_N1_a3=trajectory(n3_N1,n3_a3)

L_n3_N2_a1,gap_n3_N2_a1,entanglement_n3_N2_a1=trajectory(n3_N2,n3_a1)
L_n3_N2_a2,gap_n3_N2_a2,entanglement_n3_N2_a2=trajectory(n3_N2,n3_a2)
L_n3_N2_a3,gap_n3_N2_a3,entanglement_n3_N2_a3=trajectory(n3_N2,n3_a3)

L_n3_N3_a1,gap_n3_N3_a1,entanglement_n3_N3_a1=trajectory(n3_N3,n3_a1)
L_n3_N3_a2,gap_n3_N3_a2,entanglement_n3_N3_a2=trajectory(n3_N3,n3_a2)
L_n3_N3_a3,gap_n3_N3_a3,entanglement_n3_N3_a3=trajectory(n3_N3,n3_a3)

x=np.linspace(0.000001,0.999999,1000)
fig=plt.figure(1,figsize=(4.5,5))
ax = fig.add_subplot()

for i in np.arange(0,10*np.log(2),np.log(2)/2):
    plt.axhline(y=i,color='silver',linewidth=1,linestyle='solid',alpha=0.2)

plt.fill_between(x, f(256,8*256,x),color='skyblue',alpha=0.2)
plt.fill_between(x, f(256,8*256,x),f(512,8*512,x),color='forestgreen',alpha=0.2)
plt.fill_between(x, f(512,8*512,x),f(1024,8*1024,x),color='blueviolet',alpha=0.2)
plt.fill_between(x, -x*np.log(x)-(1-x)*np.log(1-x),color='silver')
plt.fill_between(x, -np.log(x),color='silver')
    
plt.axvline(x=0.5,color='black',linestyle='dotted')
x=np.linspace(0.500001,0.999999,1000)
plt.plot(x,-x*np.log(x)-(1-x)*np.log(x-0.5),color='black',linestyle='dashed')

x=np.linspace((1+np.sqrt(1/8))**2/1024,0.999999,1000)
plt.plot(x,f(1024,8*1024,x),'-',color='blueviolet',linewidth=1,label=''r'$\alpha=1024$')
x=np.linspace((1+np.sqrt(1/8))**2/512,0.999999,1000)
plt.plot(x,f(512,8*512,x),'-',color='forestgreen',linewidth=1,label=''r'$\alpha=512$')
x=np.linspace((1+np.sqrt(1/8))**2/256,0.999999,1000)
plt.plot(x,f(256,8*256,x),'-',color='skyblue',linewidth=1,label=''r'$\alpha=256$')

plt.plot(L_n3_N1_a1,entanglement_n3_N1_a1,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L_n3_N1_a2,entanglement_n3_N1_a2,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L_n3_N1_a3,entanglement_n3_N1_a3,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L_n3_N2_a1,entanglement_n3_N2_a1,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L_n3_N2_a2,entanglement_n3_N2_a2,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L_n3_N2_a3,entanglement_n3_N2_a3,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L_n3_N3_a1,entanglement_n3_N3_a1,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L_n3_N3_a2,entanglement_n3_N3_a2,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L_n3_N3_a3,entanglement_n3_N3_a3,'-',color='blueviolet',linewidth=1,alpha=0.1)

plt.plot(L_n3_N1_a1,entanglement_n3_N1_a1,'+',color='blueviolet',markersize=5,alpha=0.3)
plt.plot(L_n3_N1_a2,entanglement_n3_N1_a2,'+',color='blueviolet',markersize=5,alpha=0.3)
plt.plot(L_n3_N1_a3,entanglement_n3_N1_a3,'+',color='blueviolet',markersize=5,alpha=0.3)
plt.plot(L_n3_N2_a1,entanglement_n3_N2_a1,'+',color='blueviolet',markersize=5,alpha=0.3)
plt.plot(L_n3_N2_a2,entanglement_n3_N2_a2,'+',color='blueviolet',markersize=5,alpha=0.3)
plt.plot(L_n3_N2_a3,entanglement_n3_N2_a3,'+',color='blueviolet',markersize=5,alpha=0.3)
plt.plot(L_n3_N3_a1,entanglement_n3_N3_a1,'+',color='blueviolet',markersize=5,alpha=0.3)
plt.plot(L_n3_N3_a2,entanglement_n3_N3_a2,'+',color='blueviolet',markersize=5,alpha=0.3)
plt.plot(L_n3_N3_a3,entanglement_n3_N3_a3,'+',color='blueviolet',markersize=5,alpha=0.3)

plt.plot(L_n2_N1_a1,entanglement_n2_N1_a1,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L_n2_N1_a2,entanglement_n2_N1_a2,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L_n2_N1_a3,entanglement_n2_N1_a3,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L_n2_N2_a1,entanglement_n2_N2_a1,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L_n2_N2_a2,entanglement_n2_N2_a2,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L_n2_N2_a3,entanglement_n2_N2_a3,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L_n2_N3_a1,entanglement_n2_N3_a1,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L_n2_N3_a2,entanglement_n2_N3_a2,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L_n2_N3_a3,entanglement_n2_N3_a3,'-',color='forestgreen',linewidth=1,alpha=0.1)

plt.plot(L_n2_N1_a1,entanglement_n2_N1_a1,'x',color='forestgreen',markersize=5,alpha=0.3)
plt.plot(L_n2_N1_a2,entanglement_n2_N1_a2,'x',color='forestgreen',markersize=5,alpha=0.3)
plt.plot(L_n2_N1_a3,entanglement_n2_N1_a3,'x',color='forestgreen',markersize=5,alpha=0.3)
plt.plot(L_n2_N2_a1,entanglement_n2_N2_a1,'x',color='forestgreen',markersize=5,alpha=0.3)
plt.plot(L_n2_N2_a2,entanglement_n2_N2_a2,'x',color='forestgreen',markersize=5,alpha=0.3)
plt.plot(L_n2_N2_a3,entanglement_n2_N2_a3,'x',color='forestgreen',markersize=5,alpha=0.3)
plt.plot(L_n2_N3_a1,entanglement_n2_N3_a1,'x',color='forestgreen',markersize=5,alpha=0.3)
plt.plot(L_n2_N3_a2,entanglement_n2_N3_a2,'x',color='forestgreen',markersize=5,alpha=0.3)
plt.plot(L_n2_N3_a3,entanglement_n2_N3_a3,'x',color='forestgreen',markersize=5,alpha=0.3)

plt.plot(L_n1_N1_a1,entanglement_n1_N1_a1,'-',color='skyblue',linewidth=1,alpha=0.1)
plt.plot(L_n1_N1_a2,entanglement_n1_N1_a2,'-',color='skyblue',linewidth=1,alpha=0.1)
plt.plot(L_n1_N1_a3,entanglement_n1_N1_a3,'-',color='skyblue',linewidth=1,alpha=0.1)
plt.plot(L_n1_N2_a1,entanglement_n1_N2_a1,'-',color='skyblue',linewidth=1,alpha=0.1)
plt.plot(L_n1_N2_a2,entanglement_n1_N2_a2,'-',color='skyblue',linewidth=1,alpha=0.1)
plt.plot(L_n1_N2_a3,entanglement_n1_N2_a3,'-',color='skyblue',linewidth=1,alpha=0.1)
plt.plot(L_n1_N3_a1,entanglement_n1_N3_a1,'-',color='skyblue',linewidth=1,alpha=0.1)
plt.plot(L_n1_N3_a2,entanglement_n1_N3_a2,'-',color='skyblue',linewidth=1,alpha=0.1)
plt.plot(L_n1_N3_a3,entanglement_n1_N3_a3,'-',color='skyblue',linewidth=1,alpha=0.1)

plt.plot(L_n1_N1_a1,entanglement_n1_N1_a1,'*',color='skyblue',markersize=5,alpha=0.3)
plt.plot(L_n1_N1_a2,entanglement_n1_N1_a2,'*',color='skyblue',markersize=5,alpha=0.3)
plt.plot(L_n1_N1_a3,entanglement_n1_N1_a3,'*',color='skyblue',markersize=5,alpha=0.3)
plt.plot(L_n1_N2_a1,entanglement_n1_N2_a1,'*',color='skyblue',markersize=5,alpha=0.3)
plt.plot(L_n1_N2_a2,entanglement_n1_N2_a2,'*',color='skyblue',markersize=5,alpha=0.3)
plt.plot(L_n1_N2_a3,entanglement_n1_N2_a3,'*',color='skyblue',markersize=5,alpha=0.3)
plt.plot(L_n1_N3_a1,entanglement_n1_N3_a1,'*',color='skyblue',markersize=5,alpha=0.3)
plt.plot(L_n1_N3_a2,entanglement_n1_N3_a2,'*',color='skyblue',markersize=5,alpha=0.3)
plt.plot(L_n1_N3_a3,entanglement_n1_N3_a3,'*',color='skyblue',markersize=5,alpha=0.3)

plt.xlim(0,1)
plt.ylim(0,np.log(1024))
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0])
plt.xlabel(''r'$\lambda_0$')
plt.ylabel('entropy')
plt.title(''r'$\beta=8\alpha$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fancybox=True, shadow=True, ncol=5)
ax.set_box_aspect(1)
plt.tight_layout()
#FIG. 15
plt.savefig('Shor.pdf')

fig=plt.figure(2,figsize=(4.5,5))
ax = fig.add_subplot()

for i in np.arange(0,5*np.log(2),np.log(2)/2):
    plt.axhline(y=i,color='silver',linewidth=1,linestyle='solid',alpha=0.2)

x=np.linspace(0.000001,0.999999,1000)
plt.fill_between(x, -x*np.log(x)-(1-x)*np.log(1-x),color='silver')
plt.fill_between(x, -np.log(x),color='silver')

plt.axvline(x=0.5,color='black',linestyle='dotted')
x=np.linspace(0.500001,0.999999,1000)
plt.plot(x,-x*np.log(x)-(1-x)*np.log(x-0.5),color='black',linestyle='dashed')

plt.plot(L_n3_N1_a1,entanglement_n3_N1_a1,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L_n3_N1_a2,entanglement_n3_N1_a2,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L_n3_N1_a3,entanglement_n3_N1_a3,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L_n3_N2_a1,entanglement_n3_N2_a1,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L_n3_N2_a2,entanglement_n3_N2_a2,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L_n3_N2_a3,entanglement_n3_N2_a3,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L_n3_N3_a1,entanglement_n3_N3_a1,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L_n3_N3_a2,entanglement_n3_N3_a2,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L_n3_N3_a3,entanglement_n3_N3_a3,'-',color='blueviolet',linewidth=1,alpha=0.1)

plt.plot(L_n3_N1_a1,entanglement_n3_N1_a1,'+',color='blueviolet',markersize=5,alpha=0.3,label=''r'$n=10$')
plt.plot(L_n3_N1_a2,entanglement_n3_N1_a2,'+',color='blueviolet',markersize=5,alpha=0.3)
plt.plot(L_n3_N1_a3,entanglement_n3_N1_a3,'+',color='blueviolet',markersize=5,alpha=0.3)
plt.plot(L_n3_N2_a1,entanglement_n3_N2_a1,'+',color='blueviolet',markersize=5,alpha=0.3)
plt.plot(L_n3_N2_a2,entanglement_n3_N2_a2,'+',color='blueviolet',markersize=5,alpha=0.3)
plt.plot(L_n3_N2_a3,entanglement_n3_N2_a3,'+',color='blueviolet',markersize=5,alpha=0.3)
plt.plot(L_n3_N3_a1,entanglement_n3_N3_a1,'+',color='blueviolet',markersize=5,alpha=0.3)
plt.plot(L_n3_N3_a2,entanglement_n3_N3_a2,'+',color='blueviolet',markersize=5,alpha=0.3)
plt.plot(L_n3_N3_a3,entanglement_n3_N3_a3,'+',color='blueviolet',markersize=5,alpha=0.3)

plt.plot(L_n2_N1_a1,entanglement_n2_N1_a1,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L_n2_N1_a2,entanglement_n2_N1_a2,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L_n2_N1_a3,entanglement_n2_N1_a3,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L_n2_N2_a1,entanglement_n2_N2_a1,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L_n2_N2_a2,entanglement_n2_N2_a2,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L_n2_N2_a3,entanglement_n2_N2_a3,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L_n2_N3_a1,entanglement_n2_N3_a1,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L_n2_N3_a2,entanglement_n2_N3_a2,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L_n2_N3_a3,entanglement_n2_N3_a3,'-',color='forestgreen',linewidth=1,alpha=0.1)

plt.plot(L_n2_N1_a1,entanglement_n2_N1_a1,'x',color='forestgreen',markersize=5,alpha=0.3,label=''r'$n=9$')
plt.plot(L_n2_N1_a2,entanglement_n2_N1_a2,'x',color='forestgreen',markersize=5,alpha=0.3)
plt.plot(L_n2_N1_a3,entanglement_n2_N1_a3,'x',color='forestgreen',markersize=5,alpha=0.3)
plt.plot(L_n2_N2_a1,entanglement_n2_N2_a1,'x',color='forestgreen',markersize=5,alpha=0.3)
plt.plot(L_n2_N2_a2,entanglement_n2_N2_a2,'x',color='forestgreen',markersize=5,alpha=0.3)
plt.plot(L_n2_N2_a3,entanglement_n2_N2_a3,'x',color='forestgreen',markersize=5,alpha=0.3)
plt.plot(L_n2_N3_a1,entanglement_n2_N3_a1,'x',color='forestgreen',markersize=5,alpha=0.3)
plt.plot(L_n2_N3_a2,entanglement_n2_N3_a2,'x',color='forestgreen',markersize=5,alpha=0.3)
plt.plot(L_n2_N3_a3,entanglement_n2_N3_a3,'x',color='forestgreen',markersize=5,alpha=0.3)

plt.plot(L_n1_N1_a1,entanglement_n1_N1_a1,'-',color='skyblue',linewidth=1,alpha=0.1)
plt.plot(L_n1_N1_a2,entanglement_n1_N1_a2,'-',color='skyblue',linewidth=1,alpha=0.1)
plt.plot(L_n1_N1_a3,entanglement_n1_N1_a3,'-',color='skyblue',linewidth=1,alpha=0.1)
plt.plot(L_n1_N2_a1,entanglement_n1_N2_a1,'-',color='skyblue',linewidth=1,alpha=0.1)
plt.plot(L_n1_N2_a2,entanglement_n1_N2_a2,'-',color='skyblue',linewidth=1,alpha=0.1)
plt.plot(L_n1_N2_a3,entanglement_n1_N2_a3,'-',color='skyblue',linewidth=1,alpha=0.1)
plt.plot(L_n1_N3_a1,entanglement_n1_N3_a1,'-',color='skyblue',linewidth=1,alpha=0.1)
plt.plot(L_n1_N3_a2,entanglement_n1_N3_a2,'-',color='skyblue',linewidth=1,alpha=0.1)
plt.plot(L_n1_N3_a3,entanglement_n1_N3_a3,'-',color='skyblue',linewidth=1,alpha=0.1)

plt.plot(L_n1_N1_a1,entanglement_n1_N1_a1,'*',color='skyblue',markersize=5,alpha=0.3,label=''r'$n=8$')
plt.plot(L_n1_N1_a2,entanglement_n1_N1_a2,'*',color='skyblue',markersize=5,alpha=0.3)
plt.plot(L_n1_N1_a3,entanglement_n1_N1_a3,'*',color='skyblue',markersize=5,alpha=0.3)
plt.plot(L_n1_N2_a1,entanglement_n1_N2_a1,'*',color='skyblue',markersize=5,alpha=0.3)
plt.plot(L_n1_N2_a2,entanglement_n1_N2_a2,'*',color='skyblue',markersize=5,alpha=0.3)
plt.plot(L_n1_N2_a3,entanglement_n1_N2_a3,'*',color='skyblue',markersize=5,alpha=0.3)
plt.plot(L_n1_N3_a1,entanglement_n1_N3_a1,'*',color='skyblue',markersize=5,alpha=0.3)
plt.plot(L_n1_N3_a2,entanglement_n1_N3_a2,'*',color='skyblue',markersize=5,alpha=0.3)
plt.plot(L_n1_N3_a3,entanglement_n1_N3_a3,'*',color='skyblue',markersize=5,alpha=0.3)

plt.xlim(0,1)
plt.ylim(0,5*np.log(2))
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0])
plt.yticks([0,1,2,3])
plt.xlabel(''r'$\lambda_0$')
plt.ylabel('entropy')
plt.title(''r'$\alpha=2^n$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fancybox=True, shadow=True, ncol=5)
ax.set_box_aspect(1)
plt.tight_layout()
#FIG. 8
plt.savefig('Shor_bis.pdf')

fig=plt.figure(3,figsize=(4.5,5))
ax = fig.add_subplot()
x=np.linspace(0.000001,0.999999,1000)

for i in np.arange(0,20*np.log(2),np.log(2)):
    plt.axhline(y=i,color='silver',linewidth=1,linestyle='solid',alpha=0.2)

plt.fill_between(x, f_gap(256,8*256,x),color='skyblue',alpha=0.2)
plt.fill_between(x, f_gap(256,8*256,x),f_gap(512,8*512,x),color='forestgreen',alpha=0.2)
plt.fill_between(x, f_gap(512,8*512,x),f_gap(1024,8*1024,x),color='blueviolet',alpha=0.2)
plt.fill_between(x, np.log(x)-np.log(1-x),color='silver')

plt.axvline(x=0.5,color='black',linestyle='dotted')
x=np.linspace(0.500001,0.999999,1000)
plt.plot(x,np.log(x)-np.log((2*x-1)/2),color='black',linestyle='dashed')

x=np.linspace((1+np.sqrt(1/8))**2/1024,0.999999,1000)
plt.plot(x,f_gap(1024,8*1024,x),'-',color='blueviolet',linewidth=1,label=''r'$\alpha=1024$')
x=np.linspace((1+np.sqrt(1/8))**2/512,0.999999,1000)
plt.plot(x,f_gap(512,8*512,x),'-',color='forestgreen',linewidth=1,label=''r'$\alpha=512$')
x=np.linspace((1+np.sqrt(1/8))**2/256,0.999999,1000)
plt.plot(x,f_gap(256,8*256,x),'-',color='skyblue',linewidth=1,label=''r'$\alpha=256$')

plt.plot(L_n3_N1_a1,gap_n3_N1_a1,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L_n3_N1_a2,gap_n3_N1_a2,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L_n3_N1_a3,gap_n3_N1_a3,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L_n3_N2_a1,gap_n3_N2_a1,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L_n3_N2_a2,gap_n3_N2_a2,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L_n3_N2_a3,gap_n3_N2_a3,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L_n3_N3_a1,gap_n3_N3_a1,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L_n3_N3_a2,gap_n3_N3_a2,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L_n3_N3_a3,gap_n3_N3_a3,'-',color='blueviolet',linewidth=1,alpha=0.1)

plt.plot(L_n3_N1_a1,gap_n3_N1_a1,'+',color='blueviolet',markersize=5,alpha=0.3)
plt.plot(L_n3_N1_a2,gap_n3_N1_a2,'+',color='blueviolet',markersize=5,alpha=0.3)
plt.plot(L_n3_N1_a3,gap_n3_N1_a3,'+',color='blueviolet',markersize=5,alpha=0.3)
plt.plot(L_n3_N2_a1,gap_n3_N2_a1,'+',color='blueviolet',markersize=5,alpha=0.3)
plt.plot(L_n3_N2_a2,gap_n3_N2_a2,'+',color='blueviolet',markersize=5,alpha=0.3)
plt.plot(L_n3_N2_a3,gap_n3_N2_a3,'+',color='blueviolet',markersize=5,alpha=0.3)
plt.plot(L_n3_N3_a1,gap_n3_N3_a1,'+',color='blueviolet',markersize=5,alpha=0.3)
plt.plot(L_n3_N3_a2,gap_n3_N3_a2,'+',color='blueviolet',markersize=5,alpha=0.3)
plt.plot(L_n3_N3_a3,gap_n3_N3_a3,'+',color='blueviolet',markersize=5,alpha=0.3)

plt.plot(L_n2_N1_a1,gap_n2_N1_a1,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L_n2_N1_a2,gap_n2_N1_a2,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L_n2_N1_a3,gap_n2_N1_a3,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L_n2_N2_a1,gap_n2_N2_a1,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L_n2_N2_a2,gap_n2_N2_a2,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L_n2_N2_a3,gap_n2_N2_a3,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L_n2_N3_a1,gap_n2_N3_a1,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L_n2_N3_a2,gap_n2_N3_a2,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L_n2_N3_a3,gap_n2_N3_a3,'-',color='forestgreen',linewidth=1,alpha=0.1)

plt.plot(L_n2_N1_a1,gap_n2_N1_a1,'x',color='forestgreen',markersize=5,alpha=0.3)
plt.plot(L_n2_N1_a2,gap_n2_N1_a2,'x',color='forestgreen',markersize=5,alpha=0.3)
plt.plot(L_n2_N1_a3,gap_n2_N1_a3,'x',color='forestgreen',markersize=5,alpha=0.3)
plt.plot(L_n2_N2_a1,gap_n2_N2_a1,'x',color='forestgreen',markersize=5,alpha=0.3)
plt.plot(L_n2_N2_a2,gap_n2_N2_a2,'x',color='forestgreen',markersize=5,alpha=0.3)
plt.plot(L_n2_N2_a3,gap_n2_N2_a3,'x',color='forestgreen',markersize=5,alpha=0.3)
plt.plot(L_n2_N3_a1,gap_n2_N3_a1,'x',color='forestgreen',markersize=5,alpha=0.3)
plt.plot(L_n2_N3_a2,gap_n2_N3_a2,'x',color='forestgreen',markersize=5,alpha=0.3)
plt.plot(L_n2_N3_a3,gap_n2_N3_a3,'x',color='forestgreen',markersize=5,alpha=0.3)

plt.plot(L_n1_N1_a1,gap_n1_N1_a1,'-',color='skyblue',linewidth=1,alpha=0.1)
plt.plot(L_n1_N1_a2,gap_n1_N1_a2,'-',color='skyblue',linewidth=1,alpha=0.1)
plt.plot(L_n1_N1_a3,gap_n1_N1_a3,'-',color='skyblue',linewidth=1,alpha=0.1)
plt.plot(L_n1_N2_a1,gap_n1_N2_a1,'-',color='skyblue',linewidth=1,alpha=0.1)
plt.plot(L_n1_N2_a2,gap_n1_N2_a2,'-',color='skyblue',linewidth=1,alpha=0.1)
plt.plot(L_n1_N2_a3,gap_n1_N2_a3,'-',color='skyblue',linewidth=1,alpha=0.1)
plt.plot(L_n1_N3_a1,gap_n1_N3_a1,'-',color='skyblue',linewidth=1,alpha=0.1)
plt.plot(L_n1_N3_a2,gap_n1_N3_a2,'-',color='skyblue',linewidth=1,alpha=0.1)
plt.plot(L_n1_N3_a3,gap_n1_N3_a3,'-',color='skyblue',linewidth=1,alpha=0.1)

plt.plot(L_n1_N1_a1,gap_n1_N1_a1,'*',color='skyblue',markersize=5,alpha=0.3)
plt.plot(L_n1_N1_a2,gap_n1_N1_a2,'*',color='skyblue',markersize=5,alpha=0.3)
plt.plot(L_n1_N1_a3,gap_n1_N1_a3,'*',color='skyblue',markersize=5,alpha=0.3)
plt.plot(L_n1_N2_a1,gap_n1_N2_a1,'*',color='skyblue',markersize=5,alpha=0.3)
plt.plot(L_n1_N2_a2,gap_n1_N2_a2,'*',color='skyblue',markersize=5,alpha=0.3)
plt.plot(L_n1_N2_a3,gap_n1_N2_a3,'*',color='skyblue',markersize=5,alpha=0.3)
plt.plot(L_n1_N3_a1,gap_n1_N3_a1,'*',color='skyblue',markersize=5,alpha=0.3)
plt.plot(L_n1_N3_a2,gap_n1_N3_a2,'*',color='skyblue',markersize=5,alpha=0.3)
plt.plot(L_n1_N3_a3,gap_n1_N3_a3,'*',color='skyblue',markersize=5,alpha=0.3)

plt.xlim(0,1)
plt.ylim(0,2*np.log(1024))
plt.xlabel(''r'$\lambda_0$')
plt.ylabel('gap')
plt.title(''r'$\beta=8\alpha$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fancybox=True, shadow=True, ncol=5)
ax.set_box_aspect(1)
plt.tight_layout()
#FIG. 20(c)
plt.savefig('Shor_gap.pdf')

print("--- %s seconds ---" % (time.time() - start_time))

'''
N= 133
a= 16
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 19.

[Qibo 0.1.10|INFO|2023-02-16 18:23:35]: Using qibojit (numba) backend on /CPU:0
The quantum circuit measures s = 50972.

Quantum circuit outputs r = 9.

The value found for r is not even. Trying again.

------------------------------------------------------------

1.6111578169952603
N= 133
a= 58
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 19.

The quantum circuit measures s = 43692.

Quantum circuit outputs r = 3.

The value found for r is not even. Trying again.

------------------------------------------------------------

0.8675632265279809
N= 133
a= 101
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 19.

The quantum circuit measures s = 7283.

Quantum circuit outputs r = 9.

The value found for r is not even. Trying again.

------------------------------------------------------------

1.791759313157721
N= 143
a= 16
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 19.

The quantum circuit measures s = 56798.

Quantum circuit outputs r = 15.

The value found for r is not even. Trying again.

------------------------------------------------------------

1.9224833702540307
N= 143
a= 58
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 19.

The quantum circuit measures s = 46967.

Quantum circuit outputs r = 60.

Found as factors for 143:  11  and  13.

2.393745867522076
N= 143
a= 101
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 19.

The quantum circuit measures s = 48060.

Quantum circuit outputs r = 15.

The value found for r is not even. Trying again.

------------------------------------------------------------

2.047172251434728
N= 221
a= 16
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 19.

The quantum circuit measures s = 10923.

Quantum circuit outputs r = 6.

Found as factors for 221:  13  and  17.

1.24245332421308
N= 221
a= 58
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 19.

The quantum circuit measures s = 12288.

Quantum circuit outputs r = 16.

Found as factors for 221:  13  and  17.

2.282173976514877
N= 221
a= 101
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 19.

The quantum circuit measures s = 43691.

Quantum circuit outputs r = 3.

The value found for r is not even. Trying again.

------------------------------------------------------------

1.2424533242130809
N= 299
a= 37
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 21.

The quantum circuit measures s = 224411.

Quantum circuit outputs r = 132.

Found as factors for 299:  23  and  13.

2.7879745392671187
N= 299
a= 96
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 21.

The quantum circuit measures s = 95325.

Quantum circuit outputs r = 11.

The value found for r is not even. Trying again.

------------------------------------------------------------

2.238668403274182
N= 299
a= 200
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 21.

The quantum circuit measures s = 89367.

Quantum circuit outputs r = 44.

Found as factors for 299:  23  and  13.

2.2386684049288377
N= 319
a= 37
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 21.

The quantum circuit measures s = 67408.

Quantum circuit outputs r = 35.

The value found for r is not even. Trying again.

------------------------------------------------------------

2.817394798907031
N= 319
a= 96
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 21.

The quantum circuit measures s = 0.

Quantum circuit outputs r = 1.

The value found for r is not even. Trying again.

------------------------------------------------------------

2.470821205717032
N= 319
a= 200
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 21.

The quantum circuit measures s = 29960.

Quantum circuit outputs r = 35.

The value found for r is not even. Trying again.

------------------------------------------------------------

2.8173946584424665
N= 493
a= 37
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 21.

The quantum circuit measures s = 194268.

Quantum circuit outputs r = 112.

Found as factors for 493:  29  and  17.

2.705823018425755
N= 493
a= 96
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 21.

The quantum circuit measures s = 156818.

Quantum circuit outputs r = 112.

Found as factors for 493:  29  and  17.

2.705823003020789
N= 493
a= 200
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 21.

The quantum circuit measures s = 84261.

Quantum circuit outputs r = 28.

Trivial factors 1 and 493 found. Trying again.

------------------------------------------------------------

2.0126758451476863
N= 611
a= 127
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 23.

The quantum circuit measures s = 691452.

Quantum circuit outputs r = 138.

Trivial factors 1 and 611 found. Trying again.

------------------------------------------------------------

2.810200430825451
N= 611
a= 256
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 23.

The quantum circuit measures s = 972592.

Quantum circuit outputs r = 69.

The value found for r is not even. Trying again.

------------------------------------------------------------

2.7722198642903164
N= 611
a= 400
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 23.

The quantum circuit measures s = 341927.

Quantum circuit outputs r = 46.

Found as factors for 611:  47  and  13.

2.81020042884162
N= 713
a= 127
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 23.

The quantum circuit measures s = 44485.

Quantum circuit outputs r = 165.

The value found for r is not even. Trying again.

------------------------------------------------------------

3.24611989304268
N= 713
a= 256
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 23.

The quantum circuit measures s = 190650.

Quantum circuit outputs r = 11.

The value found for r is not even. Trying again.

------------------------------------------------------------

2.6512104123525244
N= 713
a= 400
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 23.

The quantum circuit measures s = 699050.

Quantum circuit outputs r = 3.

The value found for r is not even. Trying again.

------------------------------------------------------------

3.227607854634542
N= 899
a= 127
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 23.

The quantum circuit measures s = 132320.

Quantum circuit outputs r = 420.

Found as factors for 899:  31  and  29.

3.3667009285721767
N= 899
a= 256
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 23.

The quantum circuit measures s = 419430.

Quantum circuit outputs r = 5.

The value found for r is not even. Trying again.

------------------------------------------------------------

2.4055426449880217
N= 899
a= 400
  - Performing algorithm using a semiclassical iQFT.

  - Total number of qubits used: 23.

The quantum circuit measures s = 679078.

Quantum circuit outputs r = 105.

The value found for r is not even. Trying again.

------------------------------------------------------------

'''
