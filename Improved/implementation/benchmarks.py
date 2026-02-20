import numpy as np

def Ufun(x, a, k, m):
    return k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < -a)

# =========================
# F1 – F13 (scalable)
# =========================
def F1(x): return np.sum(x**2)
def F2(x): return np.sum(np.abs(x)) + np.prod(np.abs(x))
def F3(x): return sum(np.sum(x[:i+1])**2 for i in range(len(x)))
def F4(x): return np.max(np.abs(x))
def F5(x): return np.sum(100*(x[1:]-x[:-1]**2)**2 + (x[:-1]-1)**2)
def F6(x): return np.sum(np.floor(x + 0.5)**2)
def F7(x): return np.sum(np.arange(1,len(x)+1)*x**4) + np.random.rand()
def F8(x): return np.sum(-x*np.sin(np.sqrt(np.abs(x))))
def F9(x): return np.sum(x**2 - 10*np.cos(2*np.pi*x)) + 10*len(x)
def F10(x): return -20*np.exp(-0.2*np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(2*np.pi*x))) + 20 + np.e
def F11(x): return np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1,len(x)+1)))) + 1
def F12(x):
    d=len(x)
    return (np.pi/d)*(10*np.sin(np.pi*(1+(x[0]+1)/4))**2 +
           np.sum(((x[:-1]+1)/4)**2*(1+10*np.sin(np.pi*(1+(x[1:]+1)/4))**2)) +
           ((x[-1]+1)/4)**2) + np.sum(Ufun(x,10,100,4))
def F13(x):
    return 0.1*(np.sin(3*np.pi*x[0])**2 +
                np.sum((x[:-1]-1)**2*(1+np.sin(3*np.pi*x[1:])**2)) +
                (x[-1]-1)**2*(1+np.sin(2*np.pi*x[-1])**2)) + np.sum(Ufun(x,5,100,4))

# =========================
# F14 – F23 (fixed-D)
# =========================
def F14(x):
    aS=np.array([[-32,-16,0,16,32]*5,
                 [-32]*5+[-16]*5+[0]*5+[16]*5+[32]*5])
    bS=np.sum((x.reshape(-1,1)-aS)**6,axis=0)
    return (1/500+np.sum(1/(np.arange(1,26)+bS)))**(-1)

def F15(x):
    aK=np.array([.1957,.1947,.1735,.16,.0844,.0627,.0456,.0342,.0323,.0235,.0246])
    bK=1/np.array([.25,.5,1,2,4,6,8,10,12,14,16])
    return np.sum((aK-(x[0]*(bK**2+x[1]*bK)/(bK**2+x[2]*bK+x[3])))**2)

def F16(x):
    return 4*x[0]**2-2.1*x[0]**4+x[0]**6/3+x[0]*x[1]-4*x[1]**2+4*x[1]**4

def F17(x):
    return (x[1]-(5.1/(4*np.pi**2))*x[0]**2+5*x[0]/np.pi-6)**2 + 10*(1-1/(8*np.pi))*np.cos(x[0]) + 10

def F18(x):
    return (1+(x[0]+x[1]+1)**2*(19-14*x[0]+3*x[0]**2-14*x[1]+6*x[0]*x[1]+3*x[1]**2)) * \
           (30+(2*x[0]-3*x[1])**2*(18-32*x[0]+12*x[0]**2+48*x[1]-36*x[0]*x[1]+27*x[1]**2))

def F19(x):
    aH=np.array([[3,10,30],[.1,10,35],[3,10,30],[.1,10,35]])
    cH=np.array([1,1.2,3,3.2])
    pH=np.array([[.3689,.117,.2673],[.4699,.4387,.747],[.1091,.8732,.5547],[.03815,.5743,.8828]])
    return -np.sum(cH*np.exp(-np.sum(aH*(x-pH)**2,axis=1)))

def F20(x):
    aH=np.array([[10,3,17,3.5,1.7,8],[.05,10,17,.1,8,14],[3,3.5,1.7,10,17,8],[17,8,.05,10,.1,14]])
    cH=np.array([1,1.2,3,3.2])
    pH=np.array([[.1312,.1696,.5569,.0124,.8283,.5886],[.2329,.4135,.8307,.3736,.1004,.9991],
                 [.2348,.1415,.3522,.2883,.3047,.6650],[.4047,.8828,.8732,.5743,.1091,.0381]])
    return -np.sum(cH*np.exp(-np.sum(aH*(x-pH)**2,axis=1)))

def F21(x):
    aSH=np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],
                  [2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]])
    cSH=np.array([.1,.2,.2,.4,.4,.6,.3,.7,.5,.5])
    return -np.sum(1/(np.sum((x-aSH[:5])**2,axis=1)+cSH[:5]))

def F22(x):
    aSH=np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],
                  [2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]])
    cSH=np.array([.1,.2,.2,.4,.4,.6,.3,.7,.5,.5])
    return -np.sum(1/(np.sum((x-aSH[:7])**2,axis=1)+cSH[:7]))

def F23(x):
    aSH=np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],
                  [2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]])
    cSH=np.array([.1,.2,.2,.4,.4,.6,.3,.7,.5,.5])
    return -np.sum(1/(np.sum((x-aSH)**2,axis=1)+cSH))
