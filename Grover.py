import grover_hash_functions
import grover_3sat_functions
import numpy as np
from qibo import models, gates, callbacks
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time

start_time = time.time()

alpha=2**8
beta=2**10

def f(a,b,l):
    result=(1-l)*(np.log(a)-np.log(1-l)-a/(2*b))-l*np.log(l)
    return result
    
def f_gap(a,b,l):
    return np.log(l)-np.log(((1-l)/a)*(1+np.sqrt(a/b))**2)
    
def EC(control,solution,clauses):
    print('Starting Grover search for the solution of an Exact Cover instance')
    qubits = control[0]
    clauses_num = control[1]
    steps = int((np.pi / 4) * np.sqrt(2**qubits))
    print("Qubits encoding the solution: {}\n".format(qubits))
    print("Total number of qubits used:  {}\n".format(qubits + clauses_num + 1))
    print('t=',steps)
    entropy_3sat = callbacks.EntanglementEntropy([i for i in range(qubits)], compute_spectrum=True)
    q, c, ancilla, circuit = grover_3sat_functions.create_qc(qubits, clauses_num)
    circuit = grover_3sat_functions.grover(entropy_3sat, circuit, q, c, ancilla, clauses, steps)
    result = circuit(nshots=100)
    frequencies = result.frequencies(binary=True, registers=False)
    most_common_bitstring = frequencies.most_common(1)[0][0]
    print("Most common bitstring: {}\n".format(most_common_bitstring))
    if solution:
       print("Exact cover solution:  {}\n".format("".join(solution)))
    entanglement_3sat=list(np.array(entropy_3sat[:])*math.log(2))
    spectrum_3sat=entropy_3sat.spectrum
    L0_3sat=[]
    Gap_3sat=[]
    for i in range(len(spectrum_3sat)):
        List=list(np.sort(spectrum_3sat[i]))
        if len(List)==1:
           Gap_3sat.append(2*np.log(alpha))
        else:
           Gap_3sat.append(List[1]-List[0])
        L0=math.exp(-List[0])
        L0_3sat.append(L0)
    return L0_3sat,Gap_3sat,entanglement_3sat
    
def Hash(h_value,collisions):
    b = 7
    q = 4
    m = 8
    rot = [1, 2]
    constant_1 = 5
    constant_2 = 9
    h = "{0:0{bits}b}".format(h_value, bits=b)
    entropy_hash = callbacks.EntanglementEntropy([i for i in range(2*q)], compute_spectrum=True)
    if len(h) > 8:
       raise ValueError("Hash should be at maximum an 8-bit number but given value contains {} bits.".format(len(h)))
    print("Target hash: {}\n".format(h))
    print('Starting Grover search for preimages of a given hash value')
    grover_it = int(np.pi * np.sqrt((2**8) / collisions) / 4)
    print('t=',grover_it)
    result = grover_hash_functions.grover(entropy_hash, q, constant_1, constant_2, rot, h, grover_it)
    most_common = result.most_common(collisions)
    print("Solutions found:\n")
    print("Preimages:")
    for i in most_common:
        if grover_hash_functions.check_hash(q, i[0], h, constant_1, constant_2, rot):
           print("   - {}\n".format(i[0]))
        else:
           print("Incorrect preimage found, number of given collisions might not match.\n")
    print("Total iterations taken: {}\n".format(grover_it))
    entanglement_hash=list(np.array(entropy_hash[:])*math.log(2))
    spectrum_hash=entropy_hash.spectrum
    L0_hash=[]
    Gap_hash=[]
    for i in range(len(spectrum_hash)):
        List=list(np.sort(spectrum_hash[i]))
        if len(List)==1:
           Gap_hash.append(2*np.log(alpha))
        else:
           Gap_hash.append(List[1]-List[0])
        L0=math.exp(-List[0])
        L0_hash.append(L0)
    return L0_hash,Gap_hash,entanglement_hash
 

control_1=[10, 7, 4]
solution_1=['0', '0', '1', '0', '0', '1', '1', '0', '0', '1']
clauses_1=[[1,3,4],[2,3,9],[1,2,10],[2,4,10],[5,6,8],[1,4,6],[4,7,8]]

control_2=[10, 7, 2]
solution_2=['0', '0', '0', '0', '1', '0', '0', '0', '1', '0']
clauses_2=[[3,6,9],[2,8,9],[3,5,7],[1,9,10],[2,7,9],[2,3,5],[1,4,5]]

control_3=[10, 7, 4]
solution_3=['0', '1', '1', '0', '1', '0', '1', '0', '0', '0']
clauses_3=[[5,8,10],[3,8,9],[2,6,9],[3,4,10],[2,6,8],[1,5,6],[7,8,9]]

L0_3sat_1,Gap_3sat_1,entanglement_3sat_1=EC(control_1,solution_1,clauses_1)
L0_3sat_2,Gap_3sat_2,entanglement_3sat_2=EC(control_2,solution_2,clauses_2)
L0_3sat_3,Gap_3sat_3,entanglement_3sat_3=EC(control_3,solution_3,clauses_3)

h_value_1=187
collisions_1=1

h_value_2=163
collisions_2=2

h_value_3=133
collisions_3=3

L0_hash_1,Gap_hash_1,entanglement_hash_1=Hash(h_value_1,collisions_1)
L0_hash_2,Gap_hash_2,entanglement_hash_2=Hash(h_value_2,collisions_2)
L0_hash_3,Gap_hash_3,entanglement_hash_3=Hash(h_value_3,collisions_3)
        
x=np.linspace(0.000001,0.999999,1000)

fig=plt.figure(1,figsize=(4.5,5))
ax = fig.add_subplot()

plt.fill_between(x, f(alpha,beta,x),color='skyblue',alpha=0.2)
plt.fill_between(x, -x*np.log(x)-(1-x)*np.log(1-x),color='silver')
plt.fill_between(x, f(alpha,beta,x),np.log(alpha),color='silver')
plt.fill_between(x, -np.log(x),color='silver')
plt.plot(x,-x*np.log(x)-(1-x)*np.log(1-x),color='gray',linestyle='dashdot')
plt.plot(x,-np.log(x),color='gray',linestyle='dashed')
plt.plot(x,-x*np.log(x)-(1-x)*np.log((1-x)/(alpha)),color='gray',linestyle='solid')
plt.fill_between(x, 0, np.log(alpha), where=(x<(1+np.sqrt(alpha/beta))**2/alpha), color='silver')
plt.axvline(x=(1+np.sqrt(alpha/beta))**2/alpha,color='gray',linestyle='dotted')

x=np.linspace((1+np.sqrt(alpha/beta))**2/alpha,0.999999,1000)
plt.plot(x,f(alpha,beta,x),'-',color='skyblue',linewidth=1,label=''r'$\beta=1024$')

plt.plot(L0_hash_1,entanglement_hash_1,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L0_hash_1,entanglement_hash_1,'+',color='blueviolet',markersize=5,label='Hash',alpha=0.5)
plt.plot(L0_hash_2,entanglement_hash_2,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L0_hash_2,entanglement_hash_2,'x',color='blueviolet',markersize=5,alpha=0.5)
plt.plot(L0_hash_3,entanglement_hash_3,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L0_hash_3,entanglement_hash_3,'*',color='blueviolet',markersize=5,alpha=0.5)

plt.plot(L0_3sat_1,entanglement_3sat_1,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L0_3sat_1,entanglement_3sat_1,'+',color='forestgreen',markersize=5,label='EC',alpha=0.5)
plt.plot(L0_3sat_2,entanglement_3sat_2,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L0_3sat_2,entanglement_3sat_2,'x',color='forestgreen',markersize=5,alpha=0.5)
plt.plot(L0_3sat_3,entanglement_3sat_3,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L0_3sat_3,entanglement_3sat_3,'*',color='forestgreen',markersize=5,alpha=0.5)

plt.xlim(0,1)
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0])
plt.xlabel(''r'$\lambda_0$')
plt.ylabel('entropy')
plt.ylim(0,np.log(alpha))
plt.title(''r'$\alpha=256$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fancybox=True, shadow=True, ncol=5)
ax.set_box_aspect(1)
plt.tight_layout()
#FIG. 14
plt.savefig('Grover.pdf')


x=np.linspace(0.000001,0.999999,1000)
fig=plt.figure(2,figsize=(4.5,5))
ax = fig.add_subplot()

plt.fill_between(x, -x*np.log(x)-(1-x)*np.log(1-x),color='silver')
plt.fill_between(x, -x*np.log(x)-(1-x)*np.log((1-x)/(alpha-1)),np.log(alpha),color='silver')
plt.fill_between(x, -np.log(x),color='silver')
plt.plot(x,-x*np.log(x)-(1-x)*np.log(1-x),color='gray',linestyle='dashdot')
plt.plot(x,-np.log(x),color='gray',linestyle='dashed')
plt.plot(x,-x*np.log(x)-(1-x)*np.log((1-x)/(alpha)),color='gray',linestyle='solid')

plt.plot(L0_hash_1,entanglement_hash_1,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L0_hash_1,entanglement_hash_1,'+',color='blueviolet',markersize=5,label='Hash',alpha=0.5)
plt.plot(L0_hash_2,entanglement_hash_2,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L0_hash_2,entanglement_hash_2,'x',color='blueviolet',markersize=5,alpha=0.5)
plt.plot(L0_hash_3,entanglement_hash_3,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L0_hash_3,entanglement_hash_3,'*',color='blueviolet',markersize=5,alpha=0.5)

plt.plot(L0_3sat_1,entanglement_3sat_1,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L0_3sat_1,entanglement_3sat_1,'+',color='forestgreen',markersize=5,label='EC',alpha=0.5)
plt.plot(L0_3sat_2,entanglement_3sat_2,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L0_3sat_2,entanglement_3sat_2,'x',color='forestgreen',markersize=5,alpha=0.5)
plt.plot(L0_3sat_3,entanglement_3sat_3,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L0_3sat_3,entanglement_3sat_3,'*',color='forestgreen',markersize=5,alpha=0.5)

plt.xlim(0,1)
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0])
plt.xlabel(''r'$\lambda_0$')
plt.ylabel('entropy')
plt.ylim(0,np.log(alpha))
plt.title(''r'$\alpha=256$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fancybox=True, shadow=True, ncol=5)
ax.set_box_aspect(1)
plt.tight_layout()
#FIG. 5
plt.savefig('Grover_bis.pdf')

fig=plt.figure(3,figsize=(4.5,5))
ax = fig.add_subplot()
x=np.linspace(0.000001,0.999999,1000)
plt.fill_between(x, f_gap(alpha,beta,x),color='skyblue',alpha=0.3)
plt.fill_between(x, np.log(x)-np.log(1-x),color='silver')
plt.fill_between(x, f_gap(alpha,beta,x),2*np.log(alpha),color='silver')
plt.plot(x,np.log(x)-np.log(1-x),color='gray',linestyle='dashdot')
plt.plot(x,np.log(x)-np.log((1-x)/alpha),color='gray',linestyle='solid')
plt.axvline(x=(1+np.sqrt(alpha/beta))**2/alpha,color='gray',linestyle='dotted')
x=np.linspace((1+np.sqrt(alpha/beta))**2/alpha,0.999999,1000)
plt.plot(x,f_gap(alpha,beta,x),'-',color='skyblue',linewidth=1,label=''r'$\beta=1024$')

plt.plot(L0_hash_1,Gap_hash_1,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L0_hash_1,Gap_hash_1,'+',color='blueviolet',markersize=5,label='Hash',alpha=0.5)
plt.plot(L0_hash_2,Gap_hash_2,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L0_hash_2,Gap_hash_2,'x',color='blueviolet',markersize=5,alpha=0.5)
plt.plot(L0_hash_3,Gap_hash_3,'-',color='blueviolet',linewidth=1,alpha=0.1)
plt.plot(L0_hash_3,Gap_hash_3,'*',color='blueviolet',markersize=5,alpha=0.5)

plt.plot(L0_3sat_1,Gap_3sat_1,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L0_3sat_1,Gap_3sat_1,'+',color='forestgreen',markersize=5,label='EC',alpha=0.5)
plt.plot(L0_3sat_2,Gap_3sat_2,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L0_3sat_2,Gap_3sat_2,'x',color='forestgreen',markersize=5,alpha=0.5)
plt.plot(L0_3sat_3,Gap_3sat_3,'-',color='forestgreen',linewidth=1,alpha=0.1)
plt.plot(L0_3sat_3,Gap_3sat_3,'*',color='forestgreen',markersize=5,alpha=0.5)

plt.xlim(0,1)
plt.ylim(0,2*np.log(alpha))
plt.xlabel(''r'$\lambda_0$')
plt.ylabel('gap')
plt.title(''r'$\alpha=256$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fancybox=True, shadow=True, ncol=5)
ax.set_box_aspect(1)
plt.tight_layout()
#FIG. 20(b)
plt.savefig('Grover_gap.pdf')

print("--- %s seconds ---" % (time.time() - start_time))
