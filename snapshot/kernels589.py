from pathlib import Path
import os,sys,json,subprocess,numpy as np
h=Path(__file__).resolve().parent;os.environ['CUDA_VISIBLE_DEVICES']='1';sys.path.insert(0,str(h/'solver_bend_scaled'));from builder429 import build,wp,sha
from dat398 import apply_planar_lanes398
from newton._src.solvers.vbd.particle_vbd_kernels import create_edge_edge_division_plane_closest_pt,create_vertex_triangle_division_plane_closest_pt,planar_truncation_t
@wp.kernel
def fixture585(q:wp.array(dtype=wp.vec3),r:wp.array(dtype=wp.vec3),kind:int,t:wp.array(dtype=float),plane:wp.array(dtype=wp.vec3),validout:wp.array(dtype=int)):
 if kind==0:valid,n,d,eps=create_edge_edge_division_plane_closest_pt(q[0],r[0],q[1],r[1],q[2],r[2],q[3],r[3])
 else:valid,n,d,eps=create_vertex_triangle_division_plane_closest_pt(q[0],r[0],q[1],r[1],q[2],r[2],q[3],r[3])
 validout[0]=int(valid);plane[0]=n;plane[1]=d
 if valid:
  for j in range(4):
   sign=float(-1.)
   if j==0 or (kind==0 and j==1):sign=1.
   t[j]=planar_truncation_t(q[j],r[j],n*sign,d,.85,eps)
def limit_table(q,delta,floor=.00015,maxdisp=.0014025):
 # A single global factor preserves every self-separating plane path; may stall an entire stack.
 lam=1.;down=delta[:,2]<0
 if down.any():lam=min(lam,float(np.min(np.maximum(q[down,2]-floor,0)/(-delta[down,2]))))
 norm=float(np.linalg.norm(delta,axis=1).max());lam=min(lam,maxdisp/max(norm,1e-30));return max(0.,lam)

@wp.kernel
def common586(d:wp.array(dtype=wp.vec3),mass:wp.array(dtype=float),total:float,out:wp.array(dtype=wp.vec3)):
 i=wp.tid();wp.atomic_add(out,0,d[i]*(mass[i]/total))
@wp.kernel
def relative586(d:wp.array(dtype=wp.vec3),common:wp.array(dtype=wp.vec3),r:wp.array(dtype=wp.vec3)):
 i=wp.tid();r[i]=d[i]-common[0]
@wp.kernel
def verify586(rec:wp.array2d(dtype=int),q:wp.array(dtype=wp.vec3),r:wp.array(dtype=wp.vec3),t:wp.array(dtype=float),edges:wp.array2d(dtype=int),tris:wp.array2d(dtype=int),alpha:float,counts:wp.array(dtype=int),margin:wp.array(dtype=float)):
 i=wp.tid();kind=rec[i,0];a=rec[i,1];b=rec[i,2];ids=wp.vec4i(0)
 if kind==0:
  ids=wp.vec4i(edges[a,2],edges[a,3],edges[b,2],edges[b,3]);x=ids[0];y=ids[1];z=ids[2];w=ids[3]
  valid,n,d,eps=create_edge_edge_division_plane_closest_pt(q[x],r[x],q[y],r[y],q[z],r[z],q[w],r[w])
 else:
  ids=wp.vec4i(a,tris[b,0],tris[b,1],tris[b,2]);x=ids[0];y=ids[1];z=ids[2];w=ids[3]
  valid,n,d,eps=create_vertex_triangle_division_plane_closest_pt(q[x],r[x],q[y],r[y],q[z],r[z],q[w],r[w])
 if not valid:wp.atomic_add(counts,kind*3+1,1)
 else:
  wp.atomic_add(counts,kind*3,1)
  for j in range(4):
   v=ids[j];sign=float(-1.)
   if j==0 or (kind==0 and j==1):sign=1.
   initial=sign*wp.dot(n,q[v]-d);final=sign*wp.dot(n,q[v]+alpha*t[v]*r[v]-d);small=wp.min(initial,final);wp.atomic_min(margin,kind,small)
   if small < -1.e-8:wp.atomic_add(counts,kind*3+2,1)


@wp.kernel
def recombine587(common:wp.array(dtype=wp.vec3),relative:wp.array(dtype=wp.vec3),t:wp.array(dtype=float),out:wp.array(dtype=wp.vec3)):
 i=wp.tid();out[i]=common[0]+relative[i]*t[i]
@wp.kernel
def apply587(ref:wp.array(dtype=wp.vec3),proposed:wp.array(dtype=wp.vec3),alpha:float,displacement:wp.array(dtype=wp.vec3),q:wp.array(dtype=wp.vec3)):
 i=wp.tid();d=alpha*proposed[i];displacement[i]=d;q[i]=ref[i]+d

