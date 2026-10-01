"""Validated configuration for the executable TCOM reference implementation."""
from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
import yaml

ENVIRONMENT = dict(n_satellites=2000,n_planes=20,walker_f=1,n_users=100,horizon=800,dt=1.0,
    altitude_km=550.0,earth_radius_km=6371.0,inclination_deg=53.0,min_elevation_deg=25.0,
    start_utc='2026-09-27T00:00:00Z',episode_spacing_s=86400,ut1_minus_utc_s=0.0,
    polar_motion_x_arcsec=0.0,polar_motion_y_arcsec=0.0,rst_extension_s=1200,
    latitude_min_deg=30.,latitude_max_deg=45.,longitude_min_deg=-10.,longitude_max_deg=10.,
    n_beams=7,max_enabled_beams=4,beam_update_s=20,planning_points=500,planning_half_angle_deg=1.,
    candidate_cap=8,capacity=10,n_subbands=10,bandwidth_mhz=20.,carrier_ghz=12.,
    tx_power_dbw=10.,noise_dbm=-96.,tx_gain_dbi=40.,rx_gain_dbi=35.,
    tx_beamwidth_deg=2.,rx_beamwidth_deg=3.,tx_attenuation_cap_db=30.,rx_attenuation_cap_db=40.,
    sinr_threshold_db=0.,gas_db=.12,rain_db=5.,rain_probability=.1,rain_correlation='user',
    blockage_db=8.,blocked_to_clear=.1,clear_to_blocked=.005263,blockage_correlation='user',
    kappa=.6,t_ref_s=60.,r_ref_mbps=240.,c_interbeam=.1,c_intersatellite=.3,
    hysteresis=.05,peak_smoothing=.8,descriptor_mode='cox')
MODEL=dict(memory_dim=128,message_dim=128,edge_dim=6,mlp_message_width=213,
    coefficient_width=32,mlp_coefficient_width=287,kernels=10,grid_intervals=5,spline_order=3,
    residual_weight=1.0)
TRAINING=dict(seeds=[1,2,3,4,5],rounds=4,episodes_per_round=80,updates_per_round=5000,
    validation_episodes=10,test_episodes=30,exploration=[.20,.15,.10,.05],
    learning_rate=3e-4,final_learning_rate=3e-5,warmup_updates=500,weight_decay=1e-5,
    batch_sequences=4,sequence_length=64,burn_in=32,load_loss_weight=.1,
    selection_interval=1000,include_step_zero=False)

@dataclass
class Config:
    environment: dict = field(default_factory=lambda: deepcopy(ENVIRONMENT))
    model: dict = field(default_factory=lambda: deepcopy(MODEL))
    training: dict = field(default_factory=lambda: deepcopy(TRAINING))
    weights: tuple = (5.,.5,.5,1.)
    def __post_init__(self):
        for name,defaults in [('environment',ENVIRONMENT),('model',MODEL),('training',TRAINING)]:
            merged=deepcopy(defaults); merged.update(getattr(self,name)); setattr(self,name,merged)
        self.weights=tuple(float(x) for x in self.weights)
        e=self.environment
        for key in ['n_satellites','n_planes','n_users','horizon','n_beams','max_enabled_beams','candidate_cap','capacity','n_subbands','planning_points']:
            if not isinstance(e[key],int) or e[key]<1: raise ValueError(f'{key} must be a positive integer')
        if e['n_satellites'] % e['n_planes']: raise ValueError('n_satellites must be divisible by n_planes')
        if e['capacity']>e['n_subbands']: raise ValueError('capacity cannot exceed orthogonal subbands')
        if e['n_users']>e['planning_points']: raise ValueError('planning_points must cover the nested user pool')
        if e['max_enabled_beams']>e['n_beams']: raise ValueError('too many enabled beams')
        if e['dt']!=1.: raise ValueError('TCOM grid is one second; use dt=1')
        if not 0<e['min_elevation_deg']<90: raise ValueError('min_elevation_deg must be in (0,90)')
        if len(self.weights)!=4 or min(self.weights)<0: raise ValueError('four nonnegative weights required')
        for k in ['rain_correlation','blockage_correlation']:
            if e[k] not in ('region','user','link'): raise ValueError(f'unsupported {k}')
        if e['descriptor_mode'] not in ('cox','ephemeris'): raise ValueError('unknown descriptor_mode')
        for key in ['memory_dim','message_dim','edge_dim','mlp_message_width','coefficient_width','mlp_coefficient_width','kernels','grid_intervals','spline_order']:
            if self.model[key]!=MODEL[key]:
                raise ValueError(f'TCOM model requires {key}={MODEL[key]}; architecture changes need a separate implementation')
        if self.model['residual_weight']<0: raise ValueError('residual_weight must be nonnegative')
    @classmethod
    def from_yaml(cls,path):
        with open(path,encoding='utf-8') as f: obj=yaml.safe_load(f) or {}
        allowed={'environment','model','training','weights'}
        unknown=set(obj)-allowed
        if unknown: raise ValueError(f'unknown config sections: {sorted(unknown)}')
        return cls(**obj)
    def to_dict(self):
        return dict(environment=deepcopy(self.environment),model=deepcopy(self.model),training=deepcopy(self.training),weights=list(self.weights))
    def to_yaml(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        with open(path,'w',encoding='utf-8') as f: yaml.safe_dump(self.to_dict(),f,sort_keys=False)

def default_config(): return Config()

def from_yaml(path): return Config.from_yaml(path)
