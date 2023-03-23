import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
import time
import random
import math

start_time = time.time()

normal='real'

def MIN(l):
    SUM=l
    total=-l*np.log(l)
    while SUM!=1:
          t=min(l,1-SUM)
          SUM=SUM+t
          if t>0:
             total=total-t*np.log(t)
    return total

def f(a,l):
    return -1/2*l*(4*np.log(2)-1)-l*np.log(l)+(l-1)*np.log(-4*(l-1)/a)+2*np.log(2)-1/2
    
def f_gap(alpha,beta,l):
    return np.log(l)-np.log(((1-l)/alpha)*(1+np.sqrt(alpha/beta))**2)

fig=plt.figure(1,figsize=(4.5,5))
ax = fig.add_subplot()

alpha=128
x=np.linspace(0.000001,0.999999,1000)

X=np.arange(1/alpha, 0.5, 0.001)
X=list(X)

Y_min=[]
for i in range(len(X)):
    Y_min.append(MIN(X[i]))

plt.fill_between(x, f(alpha,x),color='skyblue',alpha=0.3)
plt.fill_between(x, -x*np.log(x)-(1-x)*np.log(1-x),color='silver')
plt.fill_between(x, -x*np.log(x)-(1-x)*np.log((1-x)/(alpha)),np.log(alpha),color='silver')
plt.fill_between(x, f(alpha,x),-x*np.log(x)-(1-x)*np.log((1-x)/(alpha)),color='silver')
plt.fill_between(x, -np.log(x),color='silver')
plt.fill_between(x, 0, np.log(alpha), where=(x<4/alpha), color='silver')

plt.axvline(x=4/alpha,color='gray',linestyle='dotted')
plt.plot(x,-x*np.log(x)-(1-x)*np.log(1-x),color='gray',linestyle='dashdot')
plt.plot(X,Y_min,'k-')
plt.plot(x,-np.log(x),color='gray',linestyle='dashed')
plt.plot(x,-x*np.log(x)-(1-x)*np.log((1-x)/(alpha)),color='gray',linestyle='solid')
x=np.linspace(4/alpha,0.999999,1000)
plt.plot(x,f(alpha,x),color='skyblue',linestyle='solid',label='MPD')
plt.plot(1/alpha,np.log(alpha),'rP',label='AME')
plt.plot(4/alpha,np.log(alpha)-1/2,'bo',label='average')

plt.xlim(0,1)
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0])
plt.xlabel(''r'$\lambda_0$')
plt.ylabel('entropy')
plt.ylim(0,np.log(alpha))
plt.title(''r'$\alpha=\beta=128$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fancybox=True, shadow=True, ncol=5)
ax.set_box_aspect(1)
plt.tight_layout()
#FIG. 12
plt.savefig('AME.pdf')

fig=plt.figure(2,figsize=(4.5,5))
ax = fig.add_subplot()
x=np.linspace(0.000001,0.999999,1000)
plt.fill_between(x, f_gap(alpha,128,x),color='skyblue',alpha=0.3)
plt.fill_between(x, np.log(x)-np.log(1-x),color='silver')
plt.fill_between(x, f_gap(alpha,128,x),2*np.log(alpha),color='silver')
plt.fill_between(x, 0, np.log(alpha), where=(x<4/alpha), color='silver')
plt.plot(x,np.log(x)-np.log(1-x),color='gray',linestyle='dashdot')
plt.plot(x,np.log(x)-np.log((1-x)/alpha),color='gray',linestyle='solid')
plt.axvline(x=4/alpha,color='gray',linestyle='dotted')
x=np.linspace(4/alpha,0.999999,1000)
plt.plot(x,f_gap(alpha,128,x),'-',color='skyblue',linewidth=1,label='MPD')
plt.plot(1/alpha,0,'rP',label='AME')
plt.plot(4/alpha,0,'bo',label='average')

plt.xlim(0,1)
plt.ylim(0,2*np.log(alpha))
plt.xlabel(''r'$\lambda_0$')
plt.ylabel('gap')
plt.title(''r'$\alpha=\beta=128$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fancybox=True, shadow=True, ncol=5)
ax.set_box_aspect(1)
plt.tight_layout()
#FIG. 19
plt.savefig('AME_gap.pdf')

print("--- %s seconds ---" % (time.time() - start_time))
