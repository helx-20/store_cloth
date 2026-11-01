"""Exclude only primitive pairs linked by a direct authored-source edge shorter than margin."""
import numpy as np

def maps(asset,model,count,margin=.0003):
 src=asset['source_positions'];f=asset['faces'];n=len(src);nf=len(f);edges=model.edge_indices.numpy();ne=len(edges)//count
 ue=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0);lens=np.linalg.norm(src[ue[:,0]]-src[ue[:,1]],axis=1);short=ue[lens<margin]
 vf={};ee={};incident_f=[set() for _ in range(n)];incident_e=[set() for _ in range(n)]
 for i,tri in enumerate(f):
  for v in tri:incident_f[v].add(i)
 for i,edge in enumerate(edges[:ne,2:]):
  for v in edge:incident_e[v].add(i)
 T=asset['vertices'][f];normals=np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0]);normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-30)
 original_short=short.copy();kept=[];rejected=[]
 for u,v in short:
  ids=sorted(incident_f[u]|incident_f[v]);angle=float(np.degrees(np.arccos(np.clip((normals[ids]@normals[ids].T).min(),-1,1))))
  (kept if angle<=0.001 else rejected).append([int(u),int(v),angle])
 short=np.array([x[:2] for x in kept],dtype=int).reshape(-1,2)
 for u,v in short:
  for x,y in [(u,v),(v,u)]:
   vf.setdefault(int(x),set()).update(i for i in incident_f[y] if x not in f[i])
   for i in incident_e[x]:
    ee.setdefault(i,set()).update(j for j in incident_e[y] if not set(edges[i,2:])&set(edges[j,2:]))
 vv={int(v+k*n):{int(t+k*nf) for t in ts} for k in range(count) for v,ts in vf.items() if ts}
 ev={int(i+k*ne):{int(j+k*ne) for j in js} for k in range(count) for i,js in ee.items() if js}
 return vv,ev,dict(policy='Only same-item primitivepairs linked by direct sourceedge length<0.3mm. Additionally require authored incident-face normal spread<=0.001deg (numerically coplanar authored star); heuristic localcollision exclusion, not materialglue or guarantee. No crossitem exclusions; broad topologyfilter remains0.',short_source_edges=short.tolist(),short_lengths_m=np.linalg.norm(src[short[:,0]]-src[short[:,1]],axis=1).tolist(),authored_star_normal_angle_limit_deg=0.001,rejected_short_edges=rejected,vf_pairs=sum(map(len,vv.values())),ee_directed_pairs=sum(map(len,ev.values())),items=count)
