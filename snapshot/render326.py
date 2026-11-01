"""Fixed orthographic CPU triangle rendering of every recorded physics frame."""
from pathlib import Path
import numpy as np,json,subprocess,sys,hashlib,time
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
FF=imageio_ffmpeg.get_ffmpeg_exe()
root=Path(sys.argv[1]);asset=np.load(sys.argv[2]);f=asset['faces'];n=len(asset['vertices']);W,H=960,720
right=np.array([1.,1.,0.])/np.sqrt(2);up=np.array([-.45,.45,.89302855]);up/=np.linalg.norm(up);depth=np.cross(right,up);center=np.array([0,0,.10]);scale=1400.

# Fixed smoothing groups from connected rest-fold topology, never spatial proximity.
rf=asset['vertices'][f];rn=np.cross(rf[:,1]-rf[:,0],rf[:,2]-rf[:,0]);rn/=np.maximum(np.linalg.norm(rn,axis=1)[:,None],1e-30)
incident=[[] for _ in range(n)]
for fi,tri in enumerate(f):
    for vi in tri:incident[vi].append(fi)
group=np.empty(f.shape,int);ng=0
for vi,ids in enumerate(incident):
    pending=set(ids)
    while pending:
        todo=[pending.pop()];members=[]
        while todo:
            a=todo.pop();members.append(a)
            near=[b for b in pending if len(set(f[a])&set(f[b]))==2 and rn[a]@rn[b]>.5]
            for b in near:pending.remove(b);todo.append(b)
        for fi in members:group[fi,np.where(f[fi]==vi)[0][0]]=ng
        ng+=1

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def proj(q):
    q=q-center;return np.column_stack([W/2+q@right*scale,H/2-q@up*scale])
def frame(q,k,seconds):
    im=Image.new('RGB',(W,H),'#f4f4ef');dr=ImageDraw.Draw(im)
    table=np.array([[-.3,-.3,0],[.3,-.3,0],[.3,.3,0],[-.3,.3,0]])
    dr.polygon([tuple(t) for t in proj(table)],fill='#e1dfd5')
    for z in np.arange(-.3,.301,.05):
        for ends in (np.array([[z,-.3,0],[z,.3,0]]),np.array([[-.3,z,0],[.3,z,0]])):
            dr.line([tuple(t) for t in proj(ends)],fill='#bcbbae',width=1)
    faces=np.concatenate([f+i*n for i in range(k)]);T=q[faces];norm=np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0]);norm/=np.maximum(np.linalg.norm(norm,axis=1)[:,None],1e-30)
    gids=np.concatenate([group+i*ng for i in range(k)])
    raw=np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0]);gn=np.zeros((ng*k,3))
    for corner in range(3):np.add.at(gn,gids[:,corner],raw)
    gn/=np.maximum(np.linalg.norm(gn,axis=1)[:,None],1e-30);corner_norm=gn[gids]
    shade=.55+.45*np.abs(norm@np.array([.3,-.4,.866]));palette=np.array([[71,145,186],[213,147,60],[123,173,105]])
    cols=(np.repeat(palette[:k],len(f),axis=0)*shade[:,None]).astype(np.uint8);cols[T[:,:,2].mean(1)<0]=[215,50,50];pix=proj(q);order=np.argsort(T.mean(1)@depth)
    rgb=np.array(im);zbuffer=np.full((H,W),-np.inf);z=q@depth
    for ix in order:
        ids=faces[ix];a,b,c=pix[ids]
        lo=np.maximum(np.floor(np.minimum(np.minimum(a,b),c)).astype(int),[0,0]);hi=np.minimum(np.ceil(np.maximum(np.maximum(a,b),c)).astype(int),[W-1,H-1])
        if (hi<lo).any():continue
        xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+1)+.5,np.arange(lo[1],hi[1]+1)+.5)
        den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-14:continue
        u=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
        v=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den;w=1-u-v
        zz=u*z[ids[0]]+v*z[ids[1]]+w*z[ids[2]]
        zs=zbuffer[lo[1]:hi[1]+1,lo[0]:hi[0]+1];dst=rgb[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        mask=(u>=-1e-10)&(v>=-1e-10)&(w>=-1e-10)&(zz>zs)
        zs[mask]=zz[mask]
        nn=u[:,:,None]*corner_norm[ix,0]+v[:,:,None]*corner_norm[ix,1]+w[:,:,None]*corner_norm[ix,2]
        nn/=np.maximum(np.linalg.norm(nn,axis=2)[:,:,None],1e-30)
        lit=.55+.45*np.abs(nn@np.array([.3,-.4,.866]))
        color=(palette[ix//len(f)]*lit[:,:,None]).astype(np.uint8)
        if T[ix,:,2].mean()<0:color[:]=[215,50,50]
        dst[mask]=color[mask]
    im=Image.fromarray(rgb);dr=ImageDraw.Draw(im)
    dr.rectangle((0,0,W,54),fill='#f4f4ef');dr.text((16,12),f'ROOT takeover / {root.name} / {k} item(s) / t={seconds:.2f}s / 1x playback',fill='#222222')
    dr.text((16,33),'DIAGNOSTIC: see measured audit, not a certification | fixed camera | grid = 5 cm',fill='#9b2020')
    dr.text((16,H-24),'Table shown transparent; below-table faces highlighted RED',fill='#9b2020')

    return im


cfg=json.loads((root/'config.json').read_text());assert sha(sys.argv[2])==cfg['asset_sha']
assert (root.parent/'run316.exit').read_text().strip()=='0'
p=root/'stage1.npz';s=json.loads((root/'summary.json').read_text());assert sha(p)==s['stages'][-1]['trajectory_sha']
z=np.load(p);assert z['q'].shape==(121,n,3) and np.isfinite(z['q']).all() and np.isfinite(z['v']).all();assert np.array_equal(z['faces'],f)
ff=root/'diagnostic326.mp4';assert not ff.exists();started=time.monotonic();frame(z['q'][0],1,0).save(root/'input326.png')
proc=subprocess.Popen([FF,'-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r','60','-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-tag:v','avc1','-movflags','+faststart',str(ff)],stdin=subprocess.PIPE)
peak=json.loads((root.parent/'audit317_full316.json').read_text())['peak_frame']
for j in range(1,121):
 im=frame(z['q'][j],1,j/60);proc.stdin.write(im.tobytes())
 if j in {1,10,30,60,90,120,peak}:im.save(root/f'frame326_{j}.png')
proc.stdin.close();assert proc.wait()==0
result=subprocess.run([FF,'-hide_banner','-loglevel','error','-i',str(ff),'-f','null','-'],capture_output=True)
reader=imageio_ffmpeg.read_frames(str(ff),pix_fmt='rgb24');meta=next(reader);count=sum(1 for _ in reader)
probe=subprocess.run([FF,'-hide_banner','-i',str(ff)],capture_output=True).stderr.decode();raw=ff.read_bytes();fast=raw.find(b'moov')<raw.find(b'mdat');assert result.returncode==0 and count==120 and 'avc1' in probe and 'yuv420p' in probe and fast
r=dict(asset_sha=sha(sys.argv[2]),trajectory_sha=sha(p),script_sha=sha(__file__),video_sha=sha(ff),frames=count,decode_exit=result.returncode,metadata=meta,probe=probe,faststart=fast,wall_s=time.monotonic()-started,camera=dict(center=center.tolist(),right=right.tolist(),up=up.tolist(),pixels_per_meter=scale,width=W,height=H),policy='Actual316 stage1 frames1..120 at60fps,0..2s original time; fixed orthographic camera and5cmgrid; identity physical vertices/faces, normals only smoothing, no displacement. No stability acceptance.')
(root/'render326_manifest.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
