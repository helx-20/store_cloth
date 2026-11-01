from kernels589 import *
from bend_memory780 import BendMemory
import time,resource

assert int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=8192;wp.set_device('cuda:0');a=np.load(h/'candidate758_two_local_collapses/prefold_mesh.npz');cfg=json.loads((h/'run510_canonical468/config.json').read_text())['config'];cfg['iterations']=64;cfg['radius']=.0003;cfg['self_margin']=.0006;out=h/'run839';out.mkdir(exist_ok=False);reports=[];allcalls=[];start=time.perf_counter()
assert 'include_bending=True' in (h/'builder429.py').read_text().replace(' ','')
for mode in ['common_frame']:
 m,solver,pipe,contacts,mass=build(a,1,cfg);assert solver.friction_epsilon==.01;solver.friction_epsilon=.001;cfg['friction_epsilon_effective']=.001;assert m.body_count==0;assert m.shape_count==1;assert np.allclose(m.shape_transform.numpy()[0],[0,0,-.025,0,0,0,1]);assert np.allclose(m.shape_scale.numpy()[0],[.45,.45,.025]);assert m.shape_body.numpy()[0]==-1;s0=m.state();s1=m.state();ctl=m.control();q0=a['vertices'].astype(np.float32).copy();q0[:,2]+=.05-q0[:,2].min();v0=np.zeros_like(q0)
 for s in [s0,s1]:s.particle_q.assign(q0);s.particle_qd.assign(v0)
 memory=BendMemory(m,a)
 weights=m.particle_mass.numpy().astype(float);total=float(weights.sum());weights/=total;mean=lambda x:(x*weights[:,None]).sum(0)
 original=solver._penetration_free_truncation;common=wp.zeros(1,dtype=wp.vec3);relative=wp.empty(len(q0),dtype=wp.vec3);proposed=wp.empty(len(q0),dtype=wp.vec3);tt=wp.ones(len(q0))
 def truncate(state,contact):
  ref=solver.pos_prev_collision_detection;disp=solver.particle_displacements;Q=ref.numpy();D=disp.numpy()
  if min(float(Q[:,2].min()),float((Q+D)[:,2].min()))<=.00415:
   allcalls.append({'mode':mode,'fallback':'table_envelope'});return original(state,contact)
  common.zero_();tt.fill_(1.)
  wp.launch(common586,dim=len(q0),inputs=[disp,m.particle_mass,total,common]);wp.launch(relative586,dim=len(q0),inputs=[disp,common,relative]);wp.launch(apply_planar_lanes398,dim=solver.particle_self_contact_evaluation_kernel_launch_size*8,inputs=[ref,relative,m.tri_indices,m.edge_indices,solver.trimesh_collision_info,solver.dat_conservative_bound_relaxation,tt]);wp.launch(recombine587,dim=len(q0),inputs=[common,relative,tt,proposed]);P=proposed.numpy();C=common.numpy()[0];self_bound=solver._self_contact_query_radius*solver.dat_conservative_bound_relaxation*.5;rigid_bound=solver._rigid_dat_particle_max_displacement;relative_max=float(np.linalg.norm(P-C,axis=1).max());absolute_max=float(np.linalg.norm(P,axis=1).max());alpha=min(1.,self_bound/max(relative_max,1e-30));assert alpha*relative_max<=self_bound+1e-8
  if float((Q+alpha*P)[:,2].min())<=.00415:
   allcalls.append({'mode':mode,'fallback':'recombined_table_envelope'});return original(state,contact)
  wp.launch(apply587,dim=len(q0),inputs=[ref,proposed,alpha,disp,state.particle_q]);allcalls.append({'mode':mode,'analytic_plane_clearance_m':min(float(Q[:,2].min()),float((Q+alpha*P)[:,2].min())),'self_bound':self_bound,'rigid_bound':rigid_bound,'relative_max':relative_max,'absolute_max':absolute_max,'call_index':len(allcalls),'alpha':alpha,'unscaled_COM_displacement':mean(P).tolist(),'relative_clipping_delta':mean(P-D).tolist(),'global_bound_delta':mean((alpha-1)*P).tolist(),'before_COM_displacement':mean(D).tolist(),'after_COM_displacement':mean(alpha*P).tolist(),'relative_clipped_fraction':float((tt.numpy()<1).mean())})
 if mode=='common_frame':solver._penetration_free_truncation=truncate
 tic=time.perf_counter();qs=[q0.copy()];vs=[v0.copy()];frames=[]
 for step in range(480):
  s0.clear_forces();solver.step(s0,s1,ctl,None,1/240);s0,s1=s1,s0
  memory.update(s0.particle_q.numpy(),1/240)
  if (step+1)%4==0:
   wp.synchronize();q=s0.particle_q.numpy();v=s0.particle_qd.numpy();assert np.isfinite(q).all() and np.isfinite(v).all();qs.append(q.copy());vs.append(v.copy());frames.append({'time':(step+1)/240,'com_drop_m':float(mean(q0-q)[2]),'com_vz':float(mean(v)[2]),'min_height':float(q[:,2].min())})
 memory.save(out/'bend_memory.npz')
 np.savez_compressed(out/(mode+'.npz'),q=q,v=v,q0=q0,v0=v0,qs=qs,vs=vs,times=np.arange(121)/60);reports.append({'mode':mode,'loop_wall_s':time.perf_counter()-tic,'frames':frames})
r={'bend_memory':memory.report(),'memory_sha':sha(out/'bend_memory.npz'),'memory_script_sha':sha(h/'bend_memory780.py'),'asset_sha':sha(h/'candidate758_two_local_collapses/prefold_mesh.npz'),'config':cfg,'source_neighbor_filter':solver.source_neighbor_filter_report,'results':reports,'gravity_expected_com_v':(mean(v0)+np.array([0,0,-9.81*2])).tolist(),'calls':allcalls,'process_internal_s':time.perf_counter()-start,'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'vram_peak':'unknown','limitations':'Explicit uncalibrated0.6mmcontact-thickness sensitivity on686fullfold. Existingv4sourcefilter unchanged0.3mm/coplanar; no broad newexclusions. Initialsource-local nearpairs retained, watch prestress. Initial5cm2s: static table air branch uses ALL vertex endpoint heights above .00415; every triangle and linear swept triangle stays above this plane by convexity. Self relative query bound retained. Original DAT fallback near table. Only one static box verified; no arbitrary rigid exemption, singlecloth static-table model. Safe table envelope triggers unmodified original path. Uniform epsilon .001 and64iterations from first release; bounded evolving nonfold bend restangles. Stiffness/damping, membrane,mass and self DAT unchanged. No velocity edits. Query radius bound retained. No multiitem claim. CPU readbacks diagnostic overhead; compilations included in loop if first launch.','hashes':{x:sha(h/x) for x in ['run839.py','kernels589.py','run839/common_frame.npz','candidate758_two_local_collapses/prefold_mesh.npz','builder429.py','dat398.py']}}
(h/'run839.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='calls'}))
