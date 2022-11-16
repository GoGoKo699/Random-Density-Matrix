import argparse
import grover_functions
import numpy as np
from qibo import models, gates, callbacks
import math
import matplotlib.pyplot as plt
import time

start_time = time.time()

'''
Grover search for preimages of a given hash value
Args:
h_value (int): hash value to be converted to binary string.
b (int): number of bits to be used for the hash string.
'''

def f(a,b,l):
    result=4*a*b*l*np.log(l)+ (8*a*b*np.log(2) - a**2 - b**2)*(l-1)
    result=result-4*(a*b*l - a*b)*np.log(-2*((a + b)*l - (b-a)*(1-l) - a - b)/(a*b))
    result=result- (a+b)*(b-a)*(1-l)
    return -1/4*result/(a*b)
    
h_value = 163
b = 7
collisions=2
q = 4
m = 8
rot = [1, 2]
constant_1 = 5
constant_2 = 9
h = "{0:0{bits}b}".format(h_value, bits=b)
entropy = callbacks.EntanglementEntropy([i for i in range(2*q)], compute_spectrum=True)
if len(h) > 8:
   raise ValueError("Hash should be at maximum an 8-bit number but given value contains {} bits.".format(len(h)))
print("Target hash: {}\n".format(h))
'''
measured, total_iterations = grover_functions.grover_unknown_M(entropy, q, constant_1, constant_2, rot, h)
print("Solution found in an iterative process.\n")
print("Preimage: {}\n".format(measured))
print("Total iterations taken: {}\n".format(total_iterations))
'''
grover_it = int(np.pi * np.sqrt((2**8) / collisions) / 4)
result = grover_functions.grover(entropy, q, constant_1, constant_2, rot, h, grover_it)
most_common = result.most_common(collisions)
print("Solutions found:\n")
print("Preimages:")
for i in most_common:
    if grover_functions.check_hash(q, i[0], h, constant_1, constant_2, rot):
       print("   - {}\n".format(i[0]))
    else:
       print("Incorrect preimage found, number of given collisions might not match.\n")
print("Total iterations taken: {}\n".format(grover_it))
entanglement=list(np.array(entropy[:])*math.log(2))
spectrum=entropy.spectrum
L0=[]
for i in range(len(spectrum)):
    L0.append(math.exp(-min(spectrum[i])))
        
alpha=2**8
beta=2**10

l=np.linspace(0.02,0.999,1000)

plt.figure(1,figsize=(4.5,4.5))
plt.axhline(y=np.log(alpha)-0.5,color='gray',linestyle='dashed') 
plt.plot(l,f(alpha,beta,l),'-',color='skyblue',linewidth=1,label=''r'$\beta=1024$')
plt.plot(L0,entanglement,'+',color='blueviolet',markersize=9,label='grover')
plt.xlim(-0.05,1.05)
plt.xlabel(''r'$\lambda_0$')
plt.ylabel('entropy')
plt.title(''r'$\alpha=256$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
plt.tight_layout() 
plt.savefig('FIG_8.pdf')

print("--- %s seconds ---" % (time.time() - start_time))
