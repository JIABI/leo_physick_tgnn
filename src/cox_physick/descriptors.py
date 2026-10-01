"""Cox and deterministic-ephemeris entry descriptors in physical units."""
import numpy as np

def cox_parameters(environment):
    e=environment; re=e['earth_radius_km']; rs=re+e['altitude_km']
    elev=np.deg2rad(e['min_elevation_deg'])
    alpha=np.arccos(re/rs*np.cos(elev))-elev
    rho=e['n_planes']*np.sin(alpha)
    theta=np.sqrt(398600.8/rs**3)
    beta=e['kappa']*(e['n_satellites']/e['n_planes'])*theta/(2*np.pi)
    return float(rho),float(beta)

def cox_void(rho,beta,p,window_s):
    return np.exp(-rho*(-np.expm1(-beta*np.asarray(p)*np.asarray(window_s))))

def ephemeris_void(p,count):
    p=np.asarray(p); n=np.asarray(count)
    return np.where(n==0,1.,np.power(1.-p,n))

def physical_descriptors(p,rst,common_window,sinr,load_fraction,cost,rho,beta,t_ref=60.):
    rate=rho*beta*p
    common=cox_void(rho,beta,p,common_window)
    z=np.stack([np.broadcast_to(common[:,None],rst.shape),
        np.broadcast_to(np.log1p(rate*t_ref)[:,None],rst.shape),np.log1p(rst/t_ref),
        sinr,load_fraction,cost],axis=-1)
    return z,cox_void(rho,beta,p[:,None],rst),rate
