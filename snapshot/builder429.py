
def surface_top424(lowerq,faces,qnew):
    lo=qnew[:,:2].min(0);hi=qnew[:,:2].max(0);top=0.
    for tri in lowerq[faces]:
        poly=[x.copy() for x in tri]
        for axis,bound,sign in [(0,lo[0],1),(0,hi[0],-1),(1,lo[1],1),(1,hi[1],-1)]:
            clipped=[]
            if not poly:break
            for a,b in zip(poly,poly[1:]+poly[:1]):
                da=sign*(a[axis]-bound);db=sign*(b[axis]-bound)
                if da>=0:clipped.append(a)
                if (da>=0)!=(db>=0):clipped.append(a+(b-a)*(da/(da-db)))
            poly=clipped
        if poly:top=max(top,max(float(p[2]) for p in poly))
    return top
import warp as wp
from force403 import accumulate_lanes403
from dat398 import apply_planar_lanes398
from newton._src.solvers.vbd import particle_vbd_kernels as original_kernels
_original_launch424=wp.launch
def launch424(*args,**kw):
    kernel=kw.get('kernel',args[0] if args else None)
    if kernel is original_kernels.apply_planar_truncation_parallel_by_collision:
        assert not args
        kw=dict(kw,kernel=apply_planar_lanes398,dim=kw['dim']*8)
    if kernel is original_kernels.accumulate_self_contact_force_and_hessian:
        assert not args
        kw=dict(kw,kernel=accumulate_lanes403,dim=kw['dim']*8)
    blocks={'vertex_triangle_collision_detection_kernel': 2, 'edge_colliding_edges_detection_kernel': 2}
    key=getattr(kernel,'key','')
    if key in blocks:
        assert not args
        kw=dict(kw,block_dim=blocks[key])
    return _original_launch424(*args,**kw)
wp.launch=launch424
from source_neighbor_filter_v4 import maps as source_maps
"""Direct Newton VBD experiment. Source membrane metric, authored fold bend rest."""
from pathlib import Path
import argparse,json,time,hashlib,resource,platform,subprocess,shutil
import numpy as np
import warp as wp
import newton
from newton._src.solvers.vbd import particle_vbd_kernels

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build(asset,count,cfg):
    q=asset['vertices'];f=asset['faces'];p=asset['source_positions'];n=len(q)
    b=newton.ModelBuilder(up_axis=newton.Axis.Z)
    for _ in range(count):
        b.add_cloth_mesh(pos=wp.vec3(0,0,0),rot=wp.quat_identity(),scale=1.,vel=wp.vec3(0,0,0),vertices=q.tolist(),indices=f.ravel().tolist(),density=float(asset['density']),tri_ke=cfg['tri_ke'],tri_ka=cfg['tri_ke'],tri_kd=cfg['tri_kd'],edge_ke=cfg['edge_ke'],edge_kd=cfg['edge_kd'],particle_radius=cfg['radius'])
    # Explicit compliant-crease approximation: retain flat-panel support and fixed rest angles.
    fold_mask=np.abs(np.asarray(b.edge_rest_angle))>0.1
    props=np.asarray(b.edge_bending_properties,dtype=float)
    props[fold_mask,0]=cfg['fold_edge_ke']
    props[:,1]=cfg['bend_relaxation_time_s']*props[:,0]
    b.edge_bending_properties=props.tolist()
    assert len(b.tri_indices)==len(f)*count,'Builder dropped triangles'
    # Preserve source membrane metric (coarse source, before flattening/folding).
    a=p[f[:,1]]-p[f[:,0]];c=p[f[:,2]]-p[f[:,0]]
    l=np.linalg.norm(a,axis=1);d=np.einsum('ij,ij->i',a,c)/l
    height=np.linalg.norm(np.cross(a,c),axis=1)/l
    inv=np.zeros((len(f),2,2));inv[:,0,0]=1/l;inv[:,0,1]=-d/(l*height);inv[:,1,1]=1/height
    b.tri_poses=np.tile(inv,(count,1,1)).tolist();b.tri_areas=np.tile(l*height/2,count).tolist()
    mass=np.bincount(f.ravel(),weights=np.repeat(asset['face_mass']/3,3),minlength=n)
    assert (mass>0).all();b.particle_mass=np.tile(mass,count).tolist()
    body=-1;s=newton.ModelBuilder.ShapeConfig();s.density=0.;s.ke=1e7;s.kd=.01;s.mu=.4
    b.add_shape_box(body,xform=wp.transform(p=wp.vec3(0,0,-.025),q=wp.quat_identity()),hx=.45,hy=.45,hz=.025,cfg=s)
    b.color(include_bending=True);m=b.finalize();m.set_gravity((0,0,-9.81));m.soft_contact_ke=1e5;m.soft_contact_kd=.01;m.soft_contact_mu=.3
    pipe=newton.CollisionPipeline(m,soft_contact_gap=.004,soft_contact_max=max(20000,count*len(f)*4),enable_rigid_soft_full_surface_contact=True,contact_matching='latest')
    slots=newton.solvers.SolverBase.CollisionSlot;freq=newton.solvers.SolverBase.CollisionFrequencyType
    vf_map,ee_map,filter_report=source_maps(asset,m,count)
    solver=newton.solvers.SolverVBD(m,particle_external_vertex_contact_filtering_map=vf_map,particle_external_edge_contact_filtering_map=ee_map,iterations=cfg['iterations'],particle_enable_self_contact=True,particle_self_contact_margin=cfg['radius']*2,particle_self_contact_gap=.003,particle_vertex_contact_buffer_size=cfg['vertex_buffer'],particle_edge_contact_buffer_size=cfg['edge_buffer'],particle_edge_parallel_epsilon=1e-10,particle_topological_contact_filter_threshold=0,collision_pipeline=pipe,rigid_soft_enable_dat=True,rigid_soft_contact_use_log_barrier=True,collision_frequency_type={slots.RIGID:freq.ITERATIONS,slots.SOFT_SELF_CONTACT:freq.ITERATIONS},collision_frequency={slots.RIGID:4,slots.SOFT_SELF_CONTACT:4})
    solver.source_neighbor_filter_report=filter_report
    return m,solver,pipe,solver._pipeline_contacts,mass

def main():
    ap=argparse.ArgumentParser();ap.add_argument('asset');ap.add_argument('out');ap.add_argument('--items',type=int,default=1);ap.add_argument('--frames',type=int,default=120);ap.add_argument('--substeps',type=int,default=8);ap.add_argument('--iterations',type=int,default=4);ap.add_argument('--edge-buffer',type=int,default=4096);ap.add_argument('--vertex-buffer',type=int,default=2048);ap.add_argument('--resume-stages',type=int);ap.add_argument('--resume');a=ap.parse_args()
    out=Path(a.out);out.mkdir(exist_ok=False);start=time.monotonic();wp.init();wp.set_device('cuda:0');data=np.load(a.asset);base=data['vertices'];n=len(base)
    cfg=dict(tri_ke=10000.,tri_kd=.25,edge_ke=.003,edge_kd=.0006,radius=.00015,substeps=a.substeps,iterations=a.iterations,frames=a.frames,release_height=.05,rigid_soft_dat=True,rigid_soft_log_barrier=True,static_table=True,full_surface_contact=True,self_margin=.0003,self_gap=.001,rigid_gap=.004,detect_every_iteration=False,detect_every_n_iterations=2,contact_matching='latest')
    cfg.update(bend_relaxation_time_s=.2,fold_edge_kd=.00006,fold_edge_ke=.0003,fold_rest_angle_threshold_rad=.1,topology_filter_rings=0,self_gap=.003,edge_parallel_epsilon=1e-10,vertex_buffer=a.vertex_buffer,edge_buffer=a.edge_buffer,include_bending_in_coloring=True)
    report=dict(asset_sha=sha(a.asset),script_sha=sha(__file__),config=cfg,python=platform.python_version(),numpy=np.__version__,warp=wp.__version__,newton=newton.__version__,stages=[],membrane_reference='authored flat single-sheet source metric; no rest metric reset after folding',bend_reference='fixed authored pre-fold angles; compliant prescribed folds ke0.0003 vs flat panels0.003, threshold0.1rad; nonzero bend damping c=0.2s*k, uncalibrated constitutive sensitivity test; no dynamic plastic rest update',velocity_policy='carry exact prior q/qd; no freezing or resetting lower items',solver_history='VBD solver rebuilt at each new item; previous solver internal history not transferred',gpu_before=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True))
    report['source_neighbor_filter_sha']=sha(Path(__file__).with_name('source_neighbor_filter_v4.py'))
    report['optimization424']='DAT and contact force32lanes vs4;launch dimension8x;gates399/400/403 passed;all pairs and both EE directions retained. No zero skip, no graph. Same382physics.';report['force403_sha']=sha(Path(__file__).with_name('force403.py'));report['detection_blocks424']={'vertex_triangle_collision_detection_kernel': 2, 'edge_colliding_edges_detection_kernel': 2};report['dat398_sha']=sha(Path(__file__).with_name('dat398.py'));report['material382_experiment']='Only tri_ke/tri_ka1000 to10000, damping unchanged0.25.381 found dominant compressed panel inputs near isometric; probe membrane/contact balance without more solver iterations. Uncalibrated sensitivity, not accepted.';report['performance_experiment']='337: same331 and335material,4substeps4iterations,contact redetection every2iterations; same nominal detection count/frame as335, double solverupdates;336 broadpanel compression motivates convergence probe. Contact solving remains active eachiteration; no qdreset. Not accepted.';report['filter_change_experiment']='Authored-coplanar-only exclusions; candidate77 yields no external exclusions. Inherited stage1 policy remains documented in parent; new stage uses v4. Not uniform-policy final acceptance.'
    report['source_neighbor_policy']='Same-item direct sourceedge shorter than0.3mm AND authored incident-star spread<=0.001degrees. Currentcandidate77 yields zero external exclusions. Static heuristic not dynamic safety guarantee; broad topologyfilter0, selfcontact and foldlayer freedom preserved.'
    report['solver_kernel_sha']=sha(particle_vbd_kernels.__file__)
    report['solver_kernel_path']=particle_vbd_kernels.__file__
    lowerq=np.empty((0,3));lowerv=np.empty((0,3));first_stage=1
    if a.resume:
        previous=Path(a.resume);old=json.loads((previous/'summary.json').read_text())
        assert old['asset_sha']==report['asset_sha']
        assert {k:v for k,v in old['config'].items() if k not in ('edge_buffer','vertex_buffer')}=={k:v for k,v in cfg.items() if k not in ('edge_buffer','vertex_buffer')}
        report['inherited_capacity_change']=dict(old={k:old['config'][k] for k in ['vertex_buffer','edge_buffer']},new={k:cfg[k] for k in ['vertex_buffer','edge_buffer']},reason='424 second-item interaction allocates VF1024/EE2048, physical model unchanged; monitor saved occupancy')
        if a.resume_stages is not None:
            assert 1<=a.resume_stages<=len(old['stages'])
            old['stages']=old['stages'][:a.resume_stages]
        assert all(x['frames']==a.frames and x['last']['finite'] for x in old['stages'])
        first_stage=len(old['stages'])+1;assert first_stage<=a.items
        for rec in old['stages']:
            stage=rec['stage'];file=previous/f'stage{stage}.npz';assert sha(file)==rec['trajectory_sha']
            shutil.copy2(file,out/file.name);shutil.copy2(previous/f'stage{stage}_metrics.json',out/f'stage{stage}_metrics.json')
        prior=np.load(out/f'stage{first_stage-1}.npz');lowerq=prior['q'][-1].copy();lowerv=prior['v'][-1].copy()
        report['stages']=old['stages'];report['inherited_run']=dict(path=str(previous),summary_sha=sha(previous/'summary.json'),inherited_stages=len(old['stages']),original_script_sha=old['script_sha'],original_process_wall_s=old['process_internal_wall_s'],policy='Exact saved q/qd continuation at new garment insertion; solver history rebuilt just as in uninterrupted stage transitions.')
    (out/'config.json').write_text(json.dumps(report,indent=2));report['config_sha']=sha(out/'config.json')
    for stage in range(first_stage,a.items+1):
        ts=time.monotonic();qnew=base.copy();qnew[:,:2]+=np.array([.004*(stage-1),.003*(stage-1)])
        within=(lowerq[:,:2]>=qnew[:,:2].min(0)-.001).all(1)&(lowerq[:,:2]<=qnew[:,:2].max(0)+.001).all(1)
        all_lower_faces=np.vstack([data['faces']+i*len(base) for i in range(stage-1)]) if stage>1 else np.empty((0,3),dtype=int)
        support=surface_top424(lowerq,all_lower_faces,qnew)
        qnew[:,2]+=support+.05-qnew[:,2].min();q0=np.vstack([lowerq,qnew]);v0=np.vstack([lowerv,np.zeros_like(qnew)])
        m,solver,pipe,contacts,mass=build(data,stage,cfg);s0=m.state();s1=m.state();ctl=m.control()
        for s in (s0,s1):s.particle_q.assign(q0);s.particle_qd.assign(v0)
        assert np.array_equal(s0.particle_q.numpy()[:len(lowerq)],lowerq.astype(np.float32))
        assert np.array_equal(s0.particle_qd.numpy()[:len(lowerv)],lowerv.astype(np.float32))
        report.setdefault('insertion424',[]).append(dict(stage=stage,lower_q_exact=True,lower_v_exact=True,support_top_m=support,new_bottom_m=float(qnew[:,2].min()),release_gap_m=float(qnew[:,2].min()-support),support_method='max clipped triangle height within new XY bounding rectangle; conservative local support'))
        dt=1/60/a.substeps;qs=[q0.astype(np.float32)];vs=[v0.astype(np.float32)];metrics=[];loop=time.monotonic()
        for frame in range(a.frames):
            for _ in range(a.substeps):
                s0.clear_forces();solver.step(s0,s1,ctl,None,dt);s0,s1=s1,s0
                if frame==0 and _==0:
                    np.savez_compressed(out/'first_substep.npz',q=np.array([q0.astype(np.float32),s0.particle_q.numpy()]),v=np.array([v0.astype(np.float32),s0.particle_qd.numpy()]),faces=data['faces'],mass=mass)
            wp.synchronize();q=s0.particle_q.numpy();v=s0.particle_qd.numpy();qs.append(q.copy());vs.append(v.copy())
            speeds=np.linalg.norm(v,axis=1);ke=float((np.tile(mass,stage)*speeds**2).sum()/2)
            row=dict(frame=frame+1,minz=float(q[:,2].min()),maxz=float(q[:,2].max()),speed_max=float(speeds.max()),speed_p95=float(np.quantile(speeds,.95)),kinetic_j=ke,finite=bool(np.isfinite(q).all() and np.isfinite(v).all()))
            row['rigid_soft_contacts']=int(contacts.soft_contact_count.numpy()[0]);row['contact_capacity']=int(contacts.soft_contact_max)
            info=contacts.soft_self_contact_data
            for key in ['vertex_colliding_triangles','triangle_colliding_vertices','edge_colliding_edges']:
                if getattr(info,key+'_count') is None:
                    row[key]={'status':'not allocated'};continue
                counts=getattr(info,key+'_count').numpy();cap=getattr(info,key+'_buffer_sizes').numpy()
                row[key]=dict(total=int(counts.sum()),maximum=int(counts.max()),capacity_min=int(cap.min()),overflow_rows=int((counts>cap).sum()),active_rows=int((counts>0).sum()))
            metrics.append(row)
            if frame in (0,1,9):
                np.savez_compressed(out/('early_frame'+str(frame+1)+'.npz'),q=np.array(qs),v=np.array(vs),faces=data['faces'],mass=mass)
            if frame%10==0:print(json.dumps(dict(stage=stage,loop_s=time.monotonic()-loop,**row)),flush=True)
            if not row['finite']:break
        tag='stage'+str(stage);np.savez_compressed(out/(tag+'.npz'),q=np.array(qs),v=np.array(vs),faces=data['faces'],mass=mass)
        (out/(tag+'_metrics.json')).write_text(json.dumps(metrics,indent=2))
        rec=dict(source_neighbor_filter=solver.source_neighbor_filter_report,stage=stage,frames=len(metrics),support_top_m=support,loop_wall_s=time.monotonic()-loop,stage_wall_s=time.monotonic()-ts,last=metrics[-1],trajectory_sha=sha(out/(tag+'.npz')),total_mass_kg=float(m.particle_mass.numpy().sum()))
        report['stages'].append(rec);lowerq=q.copy();lowerv=v.copy()
        (out/'summary.json').write_text(json.dumps(report,indent=2))
        if not metrics[-1]['finite']:break
    report.update(process_internal_wall_s=time.monotonic()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,gpu_after=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True))
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if any(not s['last']['finite'] for s in report['stages']):raise SystemExit(2)
if __name__=='__main__':main()
