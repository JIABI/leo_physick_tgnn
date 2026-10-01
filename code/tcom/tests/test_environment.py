import numpy as np
import pytest
from cox_physick.config import Config
from cox_physick.environment import Environment,first_free_band
from cox_physick.channel import Channel,Weather,antenna_gain_db
from cox_physick.geometry import teme_to_ecef,gmst_angle
from cox_physick.descriptors import cox_void,cox_parameters,ephemeris_void


def small_config(**kwargs):
    e=dict(n_satellites=100,n_planes=10,n_users=4,planning_points=20,horizon=3)
    e.update(kwargs); return Config(environment=e)


def test_band_mapping_retention_lowest_free_and_full():
    assert first_free_band([0,2],4)==1
    assert first_free_band([0,2],4,retained=2)==2
    assert first_free_band(range(4),4)==-1


def test_sgp4_earth_rotation_is_not_identity():
    jd=np.array([2451545.])
    p=np.array([[[6921.,0,0]]])
    q=teme_to_ecef(p,jd)
    assert np.isclose(np.linalg.norm(q),6921.)
    assert not np.allclose(p,q)
    assert np.allclose(q[0,0,:2],6921*np.array([np.cos(gmst_angle(jd)[0]),-np.sin(gmst_angle(jd)[0])]))


def test_same_tagged_receive_pointing_for_interference():
    e=small_config().environment; c=Channel(e)
    unit=np.array([[[1.,0,0],[0,1.,0]]])
    distance=np.full((1,2),550.)
    beam=np.array([[[-1.,0,0]],[[0,-1.,0]]])
    c.update(distance,unit,beam,np.ones((2,1),bool),np.zeros((1,2)))
    g,s,i=c.sinr(np.array([0]),np.array([0]),np.array([0]),np.array([0]),
                 np.array([0,1]),np.array([0,0]),np.array([0,0]),np.array([0,1]))
    # The interferer's receive off-axis loss is 40 dB, not an independent peak gain.
    assert np.isclose(i[0]/s[0],1e-4,rtol=1e-10)
    assert np.isclose(g[0],s[0]/(i[0]+c.noise_w))
    assert antenna_gain_db(1.,40.,2.,30.)==37.


def test_causal_repeat_observation_and_cost_identity(tmp_path):
    env=Environment(small_config(),seed=7); obs=env.reset()
    assert env.observe() is obs
    before=obs.z.copy()
    requests=np.where(obs.feasible.any(1),np.argmax(np.where(obs.feasible,obs.analytic,-np.inf),axis=1),-1)
    result=env.step(requests)
    assert np.array_equal(before,obs.z)
    assert result.next_observation.epoch==1
    assert len(env.weather_records)==2
    assert result.occupancy.max()<=env.e['capacity']
    assert np.all(result.failure<=result.attempt)
    K=env.K; L=np.sum(result.occupancy**2)/(env.e['capacity']*K)
    expected=5*np.mean(result.executed_sat<0)+.5*result.c_exec.mean()+.5*L-result.rate_mbps.mean()/240
    assert result.cost.mean()==pytest.approx(expected)
    assert np.all(result.rate_mbps[result.executed_sat<0]==0)
    filename=env.export_exogenous(tmp_path/'exogenous.npz')
    with np.load(filename) as z:
        assert z['satellite_ecef_km'].shape[1]==100
        assert z['beam_plan'].shape[1]==8
        assert len(z['weather_epoch_s'])==2


def test_density_nested_exogenous_inputs():
    a=Environment(small_config(n_users=2),seed=3); b=Environment(small_config(n_users=4),seed=3)
    a.reset();b.reset()
    assert np.array_equal(a.pool,b.pool)
    assert np.array_equal(a.users,b.users[:2])
    assert np.array_equal(a.weather.rain,b.weather.rain)
    assert np.array_equal(a.planner.enabled,b.planner.enabled)
    assert np.array_equal(a.planner.targets,b.planner.targets,equal_nan=True)


def test_cox_and_ephemeris_boundaries():
    rho,beta=cox_parameters(Config().environment)
    assert rho==pytest.approx(2.9419,abs=.001)
    assert cox_void(rho,beta,0,100)==1
    assert cox_void(rho,beta,1,0)==1
    assert cox_void(rho,beta,1,10000)>=np.exp(-rho)
    assert ephemeris_void(1,0)==1
    assert ephemeris_void(1,1)==0


def controlled_environment(capacity=1):
    """Deterministic admission fixture; geometry/channel behavior tested separately."""
    env=Environment(small_config(n_satellites=20,n_planes=2,n_users=3,capacity=capacity,n_subbands=2,horizon=1),seed=1)
    obs=env.reset()
    obs.sat_id[:]=-1;obs.beam_id[:]=-1;obs.feasible[:]=False
    obs.sat_id[:,0]=0;obs.beam_id[:,0]=0;obs.feasible[:,0]=True
    obs.sinr[:,0]=[3.,5.,5.]
    env.visible[:]=True;env.planner.enabled[:]=True
    env.channel.sinr=lambda users,sats,beams,bands,*args: (np.full(len(users),10.),np.ones(len(users)),np.zeros(len(users)))
    return env


def test_capacity_admission_ties_and_no_readmission():
    env=controlled_environment(); result=env.step(np.zeros(3,int))
    # Highest frozen SINR first; equal quality resolves by lowest user index.
    assert result.executed_sat.tolist()==[-1,0,-1]
    assert result.executed_subband.tolist()==[-1,0,-1]
    assert result.info['rejected'].tolist()==[True,False,True]
    assert result.occupancy[0]==1


def test_source_reservation_blocks_reuse_with_target_rejection_fallback():
    env=controlled_environment();obs=env.observe()
    env.prev_sat=np.array([0,1,-1]);env.prev_beam=np.array([0,0,-1]);env.prev_subband=np.array([0,0,-1])
    obs.prev_sat=env.prev_sat.copy();obs.prev_beam=env.prev_beam.copy()
    obs.sat_id[:]=-1;obs.beam_id[:]=-1;obs.feasible[:]=False
    for u in [0,1]:
        obs.sat_id[u,:2]=[u,1-u];obs.beam_id[u,:2]=0;obs.feasible[u,:2]=True
    obs.sat_id[2,0]=0;obs.beam_id[2,0]=0;obs.feasible[2,0]=True
    result=env.step(np.array([1,1,0]))
    assert result.executed_sat.tolist()==[0,1,-1]
    assert result.info['rejected'].all()
    assert result.attempt.tolist()==[True,True,False]
    assert not result.failure.any()


def test_execution_check_removes_failure_once_and_does_not_readmit():
    env=controlled_environment(capacity=2);calls=[]
    def check(users,sats,beams,bands,*args):
        calls.append(users.copy())
        g=np.where(users==1,0.,10.) if len(calls)==1 else np.full(len(users),12.)
        return g,np.ones(len(users)),np.zeros(len(users))
    env.channel.sinr=check
    r=env.step(np.zeros(3,int))
    assert len(calls)==2 # one joint check and one final-rate computation
    assert r.executed_sat.tolist()==[-1,-1,0]
    assert r.info['admitted'].tolist()==[False,True,True]
    assert r.info['final_sinr'][2]==12
    assert r.rate_mbps[2]==pytest.approx(20*np.log2(13))


def test_infeasible_old_beam_same_satellite_change_retains_band_priority():
    env=controlled_environment();obs=env.observe()
    env.prev_sat=np.array([0,-1,-1]);env.prev_beam=np.array([1,-1,-1]);env.prev_subband=np.array([1,-1,-1])
    obs.prev_sat=env.prev_sat.copy();obs.prev_beam=env.prev_beam.copy()
    # Old beam 1 is absent from decision set; user 0 requests feasible beam 0.
    result=env.step(np.zeros(3,int))
    assert result.executed_sat.tolist()==[0,-1,-1]
    assert result.executed_subband.tolist()==[1,-1,-1]
    assert result.info['same_satellite_priority'].tolist()==[True,False,False]


def test_disabled_prior_beam_releases_candidate_measurement_band():
    env=controlled_environment();env.prev_sat=np.array([0,0,-1]);env.prev_beam=np.array([0,1,-1]);env.prev_subband=np.array([0,1,-1])
    env.planner.enabled[0,0]=False
    assert env._candidate_band(2,0)==0
    assert env._candidate_band(1,0)==1


def test_beam_id_matching_ties_are_lexicographic():
    from cox_physick.geometry import lexicographic_assignment
    rows,cols=lexicographic_assignment(np.zeros((3,3)),np.array([0,3,6]),np.array([10,5,8]))
    got={int(c):int(r) for r,c in zip(rows,cols)}
    assert got=={1:0,2:1,0:2}


def test_earth_occulted_interferer_contributes_zero():
    e=small_config().environment;c=Channel(e)
    unit=np.array([[[1.,0,0],[0,1.,0]]]);beam=np.array([[[-1.,0,0]],[[0,-1.,0]]])
    c.update(np.full((1,2),550.),unit,beam,np.ones((2,1),bool),np.zeros((1,2)),above_horizon=np.array([[True,False]]))
    g,s,i=c.sinr(np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([1]),np.array([0]),np.array([0]),np.array([1]))
    assert i[0]==0.
    assert g[0]==pytest.approx(s[0]/c.noise_w)


def test_config_rejects_silent_architecture_changes():
    with pytest.raises(ValueError,match='mlp_message_width=213'):
        Config(model={'mlp_message_width':128})
