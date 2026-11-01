from kernels589 import *
from bend_memory780 import BendMemory
import time,resource
start=time.perf_counter();assert int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=8192
out=h/'run841';out.mkdir(exist_ok=False);a=np.load(h/'candidate758_two_local_collapses/prefold_mesh.npz');prior=np.load(h/'input841/input.npz');cfg=json.loads((h/'run510_canonical468/config.json').read_text())['config'];cfg['iterations']=64;cfg['radius']=.0003;cfg['self_margin']=.0006;n=len(a['vertices']);q0=prior['q'];v0=prior['v'];m,solver,pipe,contacts,mass=build(a,2,cfg);assert solver.friction_epsilon==.01;solver.friction_epsilon=.001;cfg['friction_epsilon_effective']=.001;s0=m.state();s1=m.state();ctl=m.control()
for s in [s0,s1]:s.particle_q.assign(q0);s.particle_qd.assign(v0)
# Preserve lower material memory as well as q/v; fresh upper has authored rest angles.
saved=np.load(h/'run839/bend_memory.npz');memory=BendMemory(m,{'source_positions':np.tile(a['source_positions'],(2,1))});ne=len(saved['edges'])
assert len(memory.edges)==2*ne
assert np.array_equal(memory.edges[:ne],saved['edges'])
upper_expected=np.where(saved['edges']>=0,saved['edges']+n,saved['edges'])
assert np.array_equal(memory.edges[ne:],upper_expected)
assert np.array_equal(memory.rest[:ne],saved['rest']) and np.array_equal(memory.rest[ne:],saved['rest'])
assert np.array_equal(memory.eligible[:ne],saved['eligible'])
for key in ['memory','dwell','sign']:getattr(memory,key)[:ne]=saved[key]
m.edge_rest_angle.assign(memory.memory.astype(np.float32))
first=np.load(h/'run839/common_frame.npz');assert np.array_equal(q0[:n],first['q']) and np.array_equal(v0[:n],first['v'])
np.savez_compressed(out/'initial_memory.npz',memory=memory.memory,dwell=memory.dwell,sign=memory.sign,rest=memory.rest)
assert np.array_equal(memory.memory[:ne],saved['memory']) and np.array_equal(memory.dwell[:ne],saved['dwell']) and np.array_equal(memory.sign[:ne],saved['sign'])
assert m.body_count==0 and m.shape_count==1 and np.allclose(m.shape_transform.numpy()[0],[0,0,-.025,0,0,0,1]) and np.allclose(m.shape_scale.numpy()[0],[.45,.45,.025]);original=solver._penetration_free_truncation;calls=[];relative=wp.empty(2*n,dtype=wp.vec3);tt=wp.ones(2*n)
def truncate(state,contact):
 ref=solver.pos_prev_collision_detection;Q=ref.numpy();D=solver.particle_displacements.numpy().copy();result=original(state,contact);baseq=state.particle_q.numpy().copy();based=solver.particle_displacements.numpy().copy();lowermax=max(float(Q[:n,2].max()),float(baseq[:n,2].max()));guard=.00415
 if min(float(Q[n:,2].min()),float((Q+D)[n:,2].min()))<=max(lowermax,0)+guard:
  calls.append({'fallback':True});return result
 C=(D[n:]*mass[:,None]).sum(0)/mass.sum();R=based.copy();R[n:]=D[n:]-C;relative.assign(R);tt.fill_(1.)
 wp.launch(apply_planar_lanes398,dim=solver.particle_self_contact_evaluation_kernel_launch_size*8,inputs=[ref,relative,m.tri_indices,m.edge_indices,solver.trimesh_collision_info,solver.dat_conservative_bound_relaxation,tt]);T=tt.numpy();P=C+T[n:,None]*R[n:];bound=solver._self_contact_query_radius*solver.dat_conservative_bound_relaxation*.5;relative_max=float(np.linalg.norm(P-C,axis=1).max());alpha=min(1.,bound/max(relative_max,1e-30));candidate=Q[n:]+alpha*P
 clearance=min(float(Q[n:,2].min()),float(candidate[:,2].min()))-max(lowermax,0)
 if clearance<=guard:calls.append({'fallback':True,'candidate_clearance':clearance});return result
 finalq=baseq.copy();finald=based.copy();finalq[n:]=candidate;finald[n:]=alpha*P;state.particle_q.assign(finalq);solver.particle_displacements.assign(finald)
 assert np.array_equal(state.particle_q.numpy()[:n],baseq[:n]) and np.array_equal(solver.particle_displacements.numpy()[:n],based[:n]);assert alpha*relative_max<=bound+1e-8
 calls.append({'fallback':False,'clearance_m':clearance,'upper_relative_query_bound':bound,'alpha':alpha,'lower_original_q_d_exact':True})
solver._penetration_free_truncation=truncate;qs=[q0];vs=[v0];loop=time.perf_counter()
for step in range(480):
 s0.clear_forces();solver.step(s0,s1,ctl,None,1/240);s0,s1=s1,s0
 memory.update(s0.particle_q.numpy(),1/240)
 if (step+1)%4==0:
  q=s0.particle_q.numpy();v=s0.particle_qd.numpy();assert np.isfinite(q).all() and np.isfinite(v).all();qs.append(q.copy());vs.append(v.copy())

memory.save(out/'bend_memory.npz')
wp.synchronize();np.savez_compressed(out/'trajectory.npz',q=q,v=v,q0=q0,v0=v0,qs=qs,vs=vs,times=np.arange(121)/60)
r={'memory':memory.report(),'lower_memory_qv_transfer_exact':True,'memory_script_sha':sha(h/'bend_memory780.py'),'memory_sha':sha(out/'bend_memory.npz'),'initial_memory_sha':sha(out/'initial_memory.npz'),'loop_wall_s':time.perf_counter()-loop,'process_internal_s':time.perf_counter()-start,'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'vram_peak':'unknown','config':cfg,'calls':calls,'source_neighbor_filter':solver.source_neighbor_filter_report,'hashes':{x:sha(h/x) for x in ['run841.py','run841/trajectory.npz','input841/input.npz','candidate758_two_local_collapses/prefold_mesh.npz','builder429.py','dat398.py','kernels589.py']},'limitations':'Two active cloths, preserved839 lowerq/v. Upper5cm above conservatively clipped localtop. Existing627 guarded upper common translation; original fallback. Collision history rebuilt; no velocity reset oflower; precrease approximation unchanged. Bounded nonfold memory updated after completed substeps, same839 parameters;64iterations/epsilon.001 from both releases, lower memory/dwell/sign inherited exactly. No wrinkle-benefit claim; not final acceptance.'}
(out/'summary.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='calls'}))
