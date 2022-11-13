import numpy as np
import sympy as sym
import pandas as pd

# Constant parameters

pi = np.pi                    # pi
m = 0.35                      # Mass
k = 5                         # Stiffness co-efficient
wn = np.sqrt(k/m)             # Natural circular frequency
Tn = 2*pi/wn                  # Natural time period
n = 11                        # No. of data points
p = pi/0.8                    # Forcing frequency
g = 1/2                       # Gamma 
b1 = 1/4                      # Beta for Avg Accn
b2 = 1/6                      # Beta for Lin Accn
F0 = 5                        # Maximum amplitude of force
U0 = F0/k                     # Static deformation
Tmax = Tn                     # Maximum time for analysis
time = np.linspace(0,Tmax,n)  # Vector for time
delta_t = Tmax/(n-1)          # Time step for numerical evaluation
xi = np.linspace(0,0.1,n)     # Range of damping ratio values

# Duhamel's integral

def duhamel(F,xi,wd,z,lim1,lim2):
    t,tau = sym.symbols('t tau')
    exp = 1/(m*wd)*F*sym.exp(-xi*wn*(t-tau))*sym.sin(wd*(t-tau))
    y = sym.integrate(exp,(tau,lim1,lim2))
    yp = y.diff(t)
    ypp = yp.diff(t)
    x = sym.lambdify(t,y)
    xp = sym.lambdify(t,yp)
    xpp = sym.lambdify(t,ypp)
    j = x(z)
    k = xp(z)
    l = xpp(z)
    return j,k,l

# After excitation ceases

def force_off(utd,vtd,xi,wd,td,z):
    t = sym.symbols('t')
    y = sym.exp(-xi*wn*(t-td))*(utd*sym.cos(wd*(t-td)) + (vtd+xi*wn*utd)/wd*sym.sin(wd*(t-td)))
    yp = y.diff(t)
    ypp = yp.diff(t)
    x = sym.lambdify(t,y)
    xp = sym.lambdify(t,yp)
    xpp = sym.lambdify(t,ypp)
    j = x(z)
    k = xp(z)
    l = xpp(z)
    return j,k,l

# Analytical solution for sinusoidal excitation

def ana_sinu(xi,wd,td,z):
    t,tau = sym.symbols('t tau')
    u = np.linspace(0,0,n)
    v = np.linspace(0,0,n)
    a = np.linspace(0,0,n)
    i = 0
    F = F0*sym.sin(p*tau)
    u_td,v_td,a_td = duhamel(F,xi,wd,td,0,t)
    while i<n and z[i]<=td:
        u[i],v[i],a[i] = duhamel(F,xi,wd,z[i],0,t)
        i += 1
    while i<n and z[i]>td:        
        u[i],v[i],a[i] = force_off(u_td,v_td,xi,wd,td,z[i])
        i += 1
    return u,v,a

# Analytical solution for triangular impulse

def ana_line(xi,wd,td,z):
    t,tau = sym.symbols('t tau')
    u = np.linspace(0,0,n)
    v = np.linspace(0,0,n)
    a = np.linspace(0,0,n)
    i = 0
    while i<n and z[i]<=td/2:
        F = 2*F0/td*tau
        u[i],v[i],a[i] = duhamel(F,xi,wd,z[i],0,t)
        i += 1
    while i<n and td/2<z[i]<=td:
        f1 = 2*F0/td*tau
        f2 = -4*F0/td*(tau-td/2)
        d1,v1,a1 = duhamel(f1,xi,wd,z[i],0,t)
        d2,v2,a2 = duhamel(f2,xi,wd,z[i],td/2,t)
        u[i],v[i],a[i] = d1+d2,v1+v2,a1+a2
        u_td1,v_td1,a_td1 = duhamel(f1,xi,wd,td,0,t)
        u_td2,v_td2,a_td2 = duhamel(f2,xi,wd,td,td/2,t)
        u_td,v_td,a_td = u_td1+u_td2,v_td1+v_td2,a_td1+a_td2
        i += 1
    while i<n and t[i]>td:
        u[i],v[i],a[i] = force_off(u_td,v_td,xi,wd,td,z[i])
    return u,v,a

# Analytical solution for step force

def ana_step(xi,wd,td,z):
    t = sym.symbols('t')
    u = np.linspace(0,0,n)
    v = np.linspace(0,0,n)
    a = np.linspace(0,0,n)
    i = 1
    u_td,v_td,a_td = duhamel(F0,xi,wd,td,0,t)
    while i<n and z[i]<=td:
        u[i],v[i],a[i] = duhamel(F0,xi,wd,z[i],0,t)
        u_td,v_td,a_td = duhamel(F0,xi,wd,td,0,t)
        i += 1
    while i<n and z[i]>td:
        u[i],v[i],a[i] = force_off(u_td,v_td,xi,wd,td,z[i])
        i += 1
    return u,v,a

# Putting all the analytical solutions in a set

as_type = [ana_sinu,ana_line,ana_step]

# Sinusoidal force input

def f_sinu(td,z):
    F = np.linspace(0,0,n)
    for i in range(0,n,1):
        if z[i] <= td:
            F[i] = F0*np.sin(p*z[i])
        else:
            F[i] = 0
    return F

# Triangular impulse input

def f_line(td,z):
    F = np.linspace(0,0,n)
    for i in range(0,n,1):
        if z[i] <= td/2:
            F[i] = 2*F0/td*z[i]
        elif td/2 < z[i] <= td:
            F[i] = 2*F0/td*(td-z[i])
        else:
            F[i] = 0
    return F

# Step force input

def f_step(td,z):
    F = np.linspace(0,0,n)
    for i in range(0,n,1):
        if 0<z[i]<td:
            F[i] = F0
        else:
            F[i] = 0
    return F

# Putting all force inputs in a list

fc_type = [f_sinu,f_line,f_step]

# Central difference method

def cdm_soln(F,c,dt):    
    u = np.linspace(0,0,n)
    v = np.linspace(0,0,n)
    a = np.linspace(0,0,n)
    u[0] = 0                              # u0
    v[0] = 0                              # v0
    a[0] = (F[0] - c*v[0] - k*u[1])/m     # a0
    u_n1 = u[0] - dt*v[0] + dt**2/2*a[0]
    K = m/dt**2 + c/(2*dt)
    A = m/dt**2 - c/(2*dt)
    B = k - 2*m/(dt**2)    
    for i in range (0,n-1,1):
        if i == 0:
            u[i+1] = (F[i] - A*u_n1 - B*u[i])/K
            v[i] = (u[i+1] - u_n1)/(2*dt)
            a[i] = (u[i+1] - 2*u[i] + u_n1)/dt**2
        else:
            u[i+1] = (F[i] - A*u[i-1] - B*u[i])/K
            v[i] = (u[i+1] - u[i-1])/(2*dt)
            a[i] = (u[i+1] - 2*u[i] + u[i-1])/dt**2
    return u,v,a
    
# Newmark method avg accn

def nma_soln(F,c,dt):
    u = np.linspace(0,0,n)
    v = np.linspace(0,0,n)
    a = np.linspace(0,0,n)
    u[0] = 0                            # u0
    v[0] = 0                            # v0
    a[0] = (F[0] - c*v[0] - k*u[0])/m   # a0
    K = k + g/b1*c/dt + 1/(b1*dt**2)*m
    X = 1/(b1*dt)*m + g/b1*c
    Y = 1/(2*b1)*m + dt*(g/(2*b1) - 1)*c
    for i in range(0,n-1,1):
        dF = F[i+1] - F[i]
        du = (dF + X*v[i] + Y*a[i])/K
        dv = g/(b1*dt)*du - g/b1*v[i] + dt*(1 - g/(2*b1))*a[i]
        da = 1/(b1*dt**2)*du - 1/(b1*dt)*v[i] - 1/(2*b1)*a[i];
        u[i+1] = u[i] + du
        v[i+1] = v[i] + dv
        a[i+1] = a[i] + da
    return u,v,a

# Newmark method linear accn

def nml_soln(F,c,dt):
    u = np.linspace(0,0,n)
    v = np.linspace(0,0,n)
    a = np.linspace(0,0,n)
    u[0] = 0                            # u0
    v[0] = 0                            # v0
    a[0] = (F[0] - c*v[0] - k*u[0])/m   # a0
    K = k + g/b2*c/dt + 1/(b2*dt**2)*m
    X = 1/(b2*dt)*m + g/b2*c
    Y = 1/(2*b2)*m + dt*(g/(2*b2) - 1)*c
    for i in range(0,n-1,1):
        dF = F[i+1] - F[i]
        du = (dF + X*v[i] + Y*a[i])/K
        dv = g/(b2*dt)*du - g/b2*v[i] + dt*(1 - g/(2*b2))*a[i]
        da = 1/(b2*dt**2)*du - 1/(b2*dt)*v[i] - 1/(2*b2)*a[i];
        u[i+1] = u[i] + du
        v[i+1] = v[i] + dv
        a[i+1] = a[i] + da
    return u,v,a

M = np.empty(shape=(12,18),dtype='object')
r = 0
for count1 in range(0,6,1):
    q = 0
    dr = xi[count1]                   # Damping ratio for this iteration
    dc = 2*dr*np.sqrt(k*m)            # Corresponding damping co-efficient 
    Wd = wn*np.sqrt(1-dr**2)        # Damped circular frequency
    Td = 2*pi/Wd                      # Damped time period
    for count2 in range(0,3,1):
        M[q,r],M[q,r+1],M[q,r+2] = as_type[count2](dr,Wd,Td,time)
        M[q+1,r],M[q+1,r+1],M[q+1,r+2] = cdm_soln(fc_type[count2](Td,time),dc,delta_t)
        M[q+2,r],M[q+2,r+1],M[q+2,r+2] = nma_soln(fc_type[count2](Td,time),dc,delta_t)
        M[q+3,r],M[q+3,r+1],M[q+3,r+2] = nml_soln(fc_type[count2](Td,time),dc,delta_t)
        q += 4
    r += 3

df = pd.DataFrame(M)

mux_r = pd.MultiIndex.from_product([['F1','F2','F3'],['Ana','CDM','NM_a','NM_l']])
mux_c = pd.MultiIndex.from_product([['dr0','dr2','dr4','dr6','dr8','dr10'],['dis','vel','acc']])
df.index = mux_r
df.columns = mux_c

df.to_pickle('D:/M Tech/NIT/Programs/LaTeX/Response_Dataframe.pkl')