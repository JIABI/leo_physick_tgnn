"""SGP4 Walker propagation and exogenous ground-point beam planning (FP64)."""
from datetime import datetime,timedelta,timezone
import numpy as np
from scipy.optimize import linear_sum_assignment
from sgp4.api import Satrec,SatrecArray,WGS72,jday

MU_WGS72=398600.8

def utc_datetime(value):
    return datetime.fromisoformat(value.replace('Z','+00:00')).astimezone(timezone.utc)

def julian_date(dt):
    return jday(dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second+dt.microsecond/1e6)

def gmst_angle(jd_ut1):
    """Vallado GMST, radians, UT1 Julian date."""
    t=(np.asarray(jd_ut1)-2451545.)/36525.
    sec=67310.54841+(876600.*3600.+8640184.812866)*t+.093104*t*t-6.2e-6*t*t*t
    return np.remainder(sec*np.pi/43200.,2*np.pi)

def teme_to_ecef(position_teme,jd_utc,ut1_minus_utc_s=0.,xp_arcsec=0.,yp_arcsec=0.):
    """TEME→PEF via GMST, then polar motion. No TEME-as-ECEF shortcut."""
    p=np.asarray(position_teme,dtype=np.float64)
    a=gmst_angle(np.asarray(jd_utc)+ut1_minus_utc_s/86400.)
    while a.ndim<p.ndim-1: a=a[...,None]
    c,s=np.cos(a),np.sin(a)
    pef=np.stack([c*p[...,0]+s*p[...,1],-s*p[...,0]+c*p[...,1],p[...,2]],axis=-1)
    xp,yp=np.deg2rad(np.array([xp_arcsec,yp_arcsec])/3600.)
    # Polar-motion rotation under the IERS small-angle convention.
    w=np.array([[np.cos(xp),np.sin(xp)*np.sin(yp),np.sin(xp)*np.cos(yp)],
                [0,np.cos(yp),-np.sin(yp)],[-np.sin(xp),np.cos(xp)*np.sin(yp),np.cos(xp)*np.cos(yp)]])
    return pef@w.T

class WalkerEphemeris:
    def __init__(self,e,episode_id=0):
        self.e=e
        self.start=utc_datetime(e['start_utc'])+timedelta(seconds=e['episode_spacing_s']*episode_id)
        self.jd0,self.fr0=julian_date(self.start)
        # Fixed mean elements at the reference epoch; all episodes propagate the same shell.
        ref=utc_datetime(e['start_utc']); j0,f0=julian_date(ref)
        epoch=j0+f0-2433281.5
        n=e['n_satellites']; planes=e['n_planes']; q=n//planes
        motion=np.sqrt(MU_WGS72/(e['earth_radius_km']+e['altitude_km'])**3)*60.
        sats=[]
        for idx in range(n):
            plane,within=divmod(idx,q)
            sat=Satrec(); sat.sgp4init(WGS72,'i',idx+1,epoch,0.,0.,0.,0.,0.,
              np.deg2rad(e['inclination_deg']),2*np.pi*(within/q+e['walker_f']*plane/n),motion,2*np.pi*plane/planes)
            sats.append(sat)
        self.satellites=SatrecArray(sats)
        self.count=int(e['horizon']+e['rst_extension_s']+1)
        seconds=np.arange(self.count,dtype=np.float64)*e['dt']
        jd=np.full(self.count,self.jd0); fr=self.fr0+seconds/86400.
        errors,r,_=self.satellites.sgp4(jd,fr)
        if np.any(errors):
            code=int(errors[errors!=0][0]); raise RuntimeError(f'SGP4 failed with status {code}')
        self.position=teme_to_ecef(r.transpose(1,0,2),jd+fr,e['ut1_minus_utc_s'],e['polar_motion_x_arcsec'],e['polar_motion_y_arcsec'])
        self.position=np.ascontiguousarray(self.position,dtype=np.float64)


def sample_ground_pool(e,rng):
    n=e['planning_points']
    sl=rng.uniform(np.sin(np.deg2rad(e['latitude_min_deg'])),np.sin(np.deg2rad(e['latitude_max_deg'])),n)
    lat=np.arcsin(sl); lon=np.deg2rad(rng.uniform(e['longitude_min_deg'],e['longitude_max_deg'],n))
    xyz=e['earth_radius_km']*np.stack([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)],axis=-1)
    return xyz

def link_geometry(users,satellites):
    ray=satellites[None,:,:]-users[:,None,:]
    distance=np.linalg.norm(ray,axis=-1)
    unit=ray/distance[...,None]
    radial=users/np.linalg.norm(users,axis=-1,keepdims=True)
    elevation=np.arcsin(np.clip(np.einsum('knc,kc->kn',unit,radial),-1.,1.))
    return distance,unit,elevation

def lexicographic_assignment(cost,row_ids,col_ids,tolerance=1e-12):
    """Minimum total angle, then ascending beam ID for ascending target ID.

    Dummy rows/columns make unequal old/new counts explicit. Equal optima are
    resolved within 1e-12 radians; no random matching perturbation is used.
    """
    cost=np.asarray(cost); nr,nc=cost.shape; size=max(nr,nc)
    padded=np.zeros((size,size));padded[:nr,:nc]=cost
    rows=list(range(size)); columns=list(range(size)); selected=[]
    order=sorted(range(nc),key=lambda j: int(col_ids[j]))+list(range(nc,size))
    for column in order:
        sub=padded[np.ix_(rows,columns)]
        rr,cc=linear_sum_assignment(sub); optimum=float(sub[rr,cc].sum())
        candidates=sorted(rows,key=lambda r: int(row_ids[r]) if r<nr else 10**9+r)
        for row in candidates:
            rs=[x for x in rows if x!=row];cs=[x for x in columns if x!=column]
            value=float(padded[row,column])
            if rs:
                block=padded[np.ix_(rs,cs)];a,b=linear_sum_assignment(block);value+=float(block[a,b].sum())
            if value<=optimum+tolerance:
                selected.append((row,column));rows.remove(row);columns.remove(column);break
        else: raise RuntimeError('unable to resolve minimum-cost beam matching')
    matched=[(r,c) for r,c in selected if r<nr and c<nc]
    return np.array([x[0] for x in matched],int),np.array([x[1] for x in matched],int)


class BeamPlanner:
    """Plans fixed Earth targets without controller state or future channel inputs."""
    def __init__(self,e,points):
        self.e=e; self.points=points
        self.targets=np.full((e['n_satellites'],e['n_beams'],3),np.nan)
        self.enabled=np.zeros((e['n_satellites'],e['n_beams']),bool)
        self.last_interval=-1
    def update(self,epoch,satellites):
        interval=int(epoch*self.e['dt']//self.e['beam_update_s'])
        if interval==self.last_interval: return
        self.last_interval=interval
        _,_,elev=link_geometry(self.points,satellites)
        cos_angle=np.cos(np.deg2rad(self.e['planning_half_angle_deg']))
        for s in range(len(satellites)):
            valid=np.flatnonzero(elev[:,s]>=np.deg2rad(self.e['min_elevation_deg']))
            selected=[]
            if len(valid):
                directions=self.points[valid]-satellites[s]
                directions/=np.linalg.norm(directions,axis=1,keepdims=True)
                covered=directions@directions.T>=cos_angle
                uncovered=np.ones(len(valid),bool)
                for _ in range(self.e['n_beams']):
                    counts=covered@uncovered.astype(np.int32)
                    # np.argmax supplies deterministic planning-point index ties.
                    best=int(np.argmax(counts))
                    if counts[best]==0: break
                    selected.append(valid[best]); uncovered[covered[best]]=False
            old=self.targets[s].copy(); self.targets[s]=np.nan; self.enabled[s]=False
            if not selected: continue
            new=self.points[selected]; old_ids=np.flatnonzero(np.isfinite(old[:,0]))
            assignments={}; remaining=set(range(len(new)))
            if len(old_ids):
                ov=old[old_ids]-satellites[s]; ov/=np.linalg.norm(ov,axis=1,keepdims=True)
                nv=new-satellites[s]; nv/=np.linalg.norm(nv,axis=1,keepdims=True)
                rr,cc=lexicographic_assignment(np.arccos(np.clip(ov@nv.T,-1,1)),old_ids,np.asarray(selected))
                for r,c in zip(rr,cc): assignments[int(c)]=int(old_ids[r]); remaining.discard(int(c))
            free=[b for b in range(self.e['n_beams']) if b not in assignments.values()]
            for c,b in zip(sorted(remaining),free): assignments[c]=b
            for rank,target in enumerate(new):
                b=assignments[rank]; self.targets[s,b]=target
                self.enabled[s,b]=rank<self.e['max_enabled_beams']
    def direction(self,satellites):
        rays=self.targets-satellites[:,None,:]
        norms=np.linalg.norm(rays,axis=-1,keepdims=True)
        return np.divide(rays,norms,out=np.zeros_like(rays),where=np.isfinite(norms)&(norms>0))
