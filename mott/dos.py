import numpy as np

def semicircle(x,t):
    return np.sqrt((2*t)**2-x**2)/(2*t**2*np.pi)

if __name__=='__main__':
    from scipy.integrate import simpson

    t = 0.5
    oms = np.linspace(-2*t,2*t,1000)
    print(simpson(semicircle(oms,t),oms))
