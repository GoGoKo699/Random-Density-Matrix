import matplotlib.pyplot as plt
import numpy as np

fig=plt.figure(1,figsize=(4.5,5))
ax = fig.add_subplot()

alpha=32
x=np.linspace(0.000001,0.999999,1000)

plt.fill_between(x, -x*np.log(x)-(1-x)*np.log(1-x),color='silver')
plt.fill_between(x, -np.log(x),color='silver')
plt.fill_between(x, -x*np.log(x)-(1-x)*np.log((1-x)/alpha),np.log(alpha),color='silver')
plt.plot(x,-x*np.log(x)-(1-x)*np.log(1-x),color='gray',linestyle='dashdot')
plt.plot(x,-np.log(x),color='gray',linestyle='dashed')
plt.plot(x,(1-x)*np.log(alpha)-x*np.log(x)-(1-x)*np.log(1-x),color='gray',linestyle='solid')

l=np.arange(1/alpha, 0.999, 0.001)
X=list(l)

def MIN(l):
    SUM=l
    total=-l*np.log(l)
    while SUM!=1:
          t=min(l,1-SUM)
          SUM=SUM+t
          if t>0:
             total=total-t*np.log(t)
    return total
    
def MAX(l):
    return -l*np.log(l)-(1-l)*np.log((1-l)/(alpha-1))
    
Y_min=[]
for i in range(len(X)):
    Y_min.append(MIN(X[i]))
 
Y_max=[]
for i in range(len(X)):
    Y_max.append(MAX(X[i]))
    
plt.plot(X,Y_min,'k-', label='numerical boundary')
plt.plot(X,Y_max,'k-')
plt.xlim(0,1)
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0])
plt.xlabel(''r'$\lambda_0$')
plt.ylabel('entropy')
plt.ylim(0,np.log(alpha))
plt.title(''r'$\alpha=32$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fancybox=True, shadow=True, ncol=5)
ax.set_box_aspect(1)
plt.tight_layout()
#FIG. 1
plt.savefig('bounds.pdf')
