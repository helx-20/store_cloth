from pathlib import Path
import sys,json,time,resource,subprocess
h=Path(__file__).resolve().parent;out=h/'view920';out.mkdir(exist_ok=False);sys.argv=['view920',str(out),str(h/'candidate758_two_local_collapses/prefold_mesh.npz')]
s=(h/'render326.py').read_text().split('cfg=json.loads')[0];exec(s);started=time.monotonic();zs=[np.load(h/f'run{k}/trajectory.npz') for k in [841,844,849]]
assert np.array_equal(zs[0]['q'],zs[1]['q0']) and np.array_equal(zs[1]['q'],zs[2]['q0'])
assert all(np.array_equal(zs[i]['v'],zs[i+1]['v0']) for i in [0,1])
configs=[json.loads((h/f'run{k}/summary.json').read_text())['config'] for k in [841,844,849]]
keys=['tri_ke','tri_kd','edge_ke','edge_kd','fold_edge_ke','bend_relaxation_time_s','iterations','radius','self_margin','friction_epsilon_effective']
assert all(all(c[k]==configs[0][k] for k in keys) for c in configs)
ff=out/'diagnostic920.mp4';proc=subprocess.Popen([FF,'-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r','60','-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-tag:v','avc1','-movflags','+faststart',str(ff)],stdin=subprocess.PIPE)
frame(zs[0]['qs'][0],2,0).save(out/'input920.png')
for j in range(1,361):
 segment=(j-1)//120;k=(j-1)%120+1;im=frame(zs[segment]['qs'][k],2,j/60);dr=ImageDraw.Draw(im)
 dr.text((16,65),'Uniform parameters: 64 iterations / friction epsilon 1 mm/s / bounded bend memory',fill='#9b2020');dr.text((16,82),'State preserved; solver history rebuilt at 2s and 4s. Not final acceptance.',fill='#9b2020')
 proc.stdin.write(im.tobytes())
 if j in [1,30,120,121,240,241,300,360]:im.save(out/f'frame920_{j}.png')
proc.stdin.close();assert proc.wait()==0
result=subprocess.run([FF,'-hide_banner','-loglevel','error','-i',str(ff),'-f','null','-'],capture_output=True);reader=imageio_ffmpeg.read_frames(str(ff),pix_fmt='rgb24');meta=next(reader);count=sum(1 for _ in reader);probe=subprocess.run([FF,'-hide_banner','-i',str(ff)],capture_output=True).stderr.decode();raw=ff.read_bytes();fast=raw.find(b'moov')<raw.find(b'mdat');assert result.returncode==0 and count==360 and 'avc1' in probe and 'yuv420p' in probe and fast
r=dict(hashes={p:sha(h/p) for p in ['render920.py','run841/trajectory.npz','run844/trajectory.npz','run849/trajectory.npz','view920/diagnostic920.mp4','candidate758_two_local_collapses/prefold_mesh.npz','run841/summary.json','run844/summary.json','run849/summary.json','render326.py','bend_memory780.py']},frames=count,decode_exit=result.returncode,faststart=fast,metadata=meta,wall_s=time.monotonic()-started,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,camera=dict(center=center.tolist(),right=right.tolist(),up=up.tolist(),pixels_per_meter=scale,width=W,height=H),limitations='Uniform physical/numerical parameters across841/844/849; exact q/v joins but collision/solver history rebuilt at2/4s. Uncalibrated bounded-memory approximation, no proven wrinkle benefit. Actual mesh no coordinate edits. Need root visual review before delivery; not final contact/strain certification.')
(out/'manifest920.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))

