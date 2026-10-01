"""Physical link powers with one tagged receive pointing and explicit weather."""
import numpy as np

class Weather:
    def __init__(self,e,seed):
        self.e=e; self.rng=np.random.default_rng(seed)
        def shape(kind):
            return {'region':(1,1),'user':(e['planning_points'],1),'link':(e['planning_points'],e['n_satellites'])}[kind]
        self.rain=self.rng.random(shape(e['rain_correlation']))<e['rain_probability']
        stationary=e['clear_to_blocked']/(e['clear_to_blocked']+e['blocked_to_clear'])
        self.blocked=self.rng.random(shape(e['blockage_correlation']))<stationary
    def advance(self):
        u=self.rng.random(self.blocked.shape)
        self.blocked=np.where(self.blocked,u>=self.e['blocked_to_clear'],u<self.e['clear_to_blocked'])
    def attenuation(self):
        e=self.e
        rain=np.broadcast_to(self.rain,(e['planning_points'],e['n_satellites']))[:e['n_users']]
        block=np.broadcast_to(self.blocked,(e['planning_points'],e['n_satellites']))[:e['n_users']]
        return e['gas_db']+e['rain_db']*rain+e['blockage_db']*block

def antenna_gain_db(angle_deg,peak_db,beamwidth_deg,cap_db):
    return peak_db-np.minimum(12.*(np.asarray(angle_deg)/beamwidth_deg)**2,cap_db)

class Channel:
    def __init__(self,e):
        self.e=e; self.noise_w=10.**((e['noise_dbm']-30.)/10.)
    def update(self,distance_km,unit_to_satellite,beam_directions,enabled,attenuation_db,above_horizon=None):
        self.distance=distance_km; self.unit=unit_to_satellite
        self.beam_directions=beam_directions; self.enabled=enabled; self.attenuation=attenuation_db
        self.above_horizon=np.ones(distance_km.shape,bool) if above_horizon is None else np.asarray(above_horizon,bool)
    def transmit_received_power(self,users,sats,beams,receive_pointing):
        """Every signal/interferer uses the SAME receiver direction per tagged link."""
        e=self.e
        users,sats,beams=np.broadcast_arrays(users,sats,beams)
        ray=self.unit[users,sats]
        txcos=np.sum(-ray*self.beam_directions[sats,beams],axis=-1)
        rxcos=np.sum(ray*receive_pointing,axis=-1)
        txang=np.rad2deg(np.arccos(np.clip(txcos,-1.,1.)))
        rxang=np.rad2deg(np.arccos(np.clip(rxcos,-1.,1.)))
        gains=antenna_gain_db(txang,e['tx_gain_dbi'],e['tx_beamwidth_deg'],e['tx_attenuation_cap_db'])
        gains+=antenna_gain_db(rxang,e['rx_gain_dbi'],e['rx_beamwidth_deg'],e['rx_attenuation_cap_db'])
        wavelength=.299792458/e['carrier_ghz']
        path=(wavelength/(4*np.pi*self.distance[users,sats]*1000.))**2
        return 10.**((e['tx_power_dbw']+gains-self.attenuation[users,sats])/10.)*path*self.enabled[sats,beams]*self.above_horizon[users,sats]
    def sinr(self,users,sats,beams,bands,tx_sats,tx_beams,tx_bands,tx_users=None,exclude_own=True):
        users=np.asarray(users,dtype=int); sats=np.asarray(sats,dtype=int); beams=np.asarray(beams,dtype=int); bands=np.asarray(bands,dtype=int)
        pointing=self.unit[users,sats]
        signal=self.transmit_received_power(users,sats,beams,pointing)
        interference=np.zeros(len(users),dtype=np.float64)
        tx_sats=np.asarray(tx_sats); tx_beams=np.asarray(tx_beams); tx_bands=np.asarray(tx_bands)
        if tx_users is None: tx_users=np.arange(len(tx_sats))
        for band in np.unique(bands):
            ii=np.flatnonzero(bands==band)
            jj=np.flatnonzero((tx_bands==band)&(tx_sats>=0))
            if not len(jj): continue
            # Bound peak intermediate memory for dense diagnostic evaluations.
            for start in range(0,len(ii),1024):
                rows=ii[start:start+1024]
                power=self.transmit_received_power(users[rows,None],tx_sats[jj][None,:],tx_beams[jj][None,:],pointing[rows,None,:])
                mask=tx_sats[jj][None,:]!=sats[rows,None]
                if exclude_own: mask &= np.asarray(tx_users)[jj][None,:]!=users[rows,None]
                interference[rows]=np.sum(np.where(mask,power,0.),axis=1)
        return signal/(interference+self.noise_w),signal,interference
