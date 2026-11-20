# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# 梁艺萌
# 树干 / 树枝 / 树叶 三维点云分离识别 —— 项目源代码
# 未经本人允许不得商用。third_party/ 内组件许可归其原作者，见 NOTICE.txt。
# ------------------------------------------------------------------
"""Build self-contained interactive 3D viewer (three.js) for predicted trees."""
import os, sys, glob, json, base64
import numpy as np

OUT = r"E:\Desktop\场景\output"
THREE = open(r"E:\Desktop\场景\treeseg\three.min.js", encoding="utf-8").read()
COL = np.array([[46, 204, 113], [52, 120, 246], [231, 76, 60]], np.float32) / 255.0

def main(pattern="pred_test_*.npz", max_per_tree=25000, out_name=None):
    max_per_tree = int(max_per_tree)
    files = sorted(glob.glob(os.path.join(OUT, pattern)))
    trees = []
    for f in files:
        tid = os.path.basename(f).split("_")[-1].split(".")[0]
        d = np.load(f)
        c, lab = d["coord"], d["pred"]
        rng = np.random.default_rng(int(tid))
        sel = rng.choice(len(c), min(max_per_tree, len(c)), replace=False)
        xyz = c[sel].astype("<f4")
        rgb = (COL[lab[sel]] * 255).astype(np.uint8)
        buf = np.empty(len(sel), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        buf["p"] = xyz; buf["c"] = rgb
        trees.append(dict(id=tid, data=base64.b64encode(buf.tobytes()).decode(),
                          n=int(len(c)), counts=[int((lab == k).sum()) for k in range(3)]))
    html = TEMPLATE.replace("__THREE__", THREE).replace("__DATA__", json.dumps(trees))
    key = out_name or (pattern.split("_")[1].split("*")[0] if "_" in pattern else "test")
    path = os.path.join(OUT, f"viewer_{key}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print("saved", path, round(os.path.getsize(path) / 1e6, 1), "MB")

TEMPLATE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Tree trunk/branch/leaf 3D</title>
<style>body{margin:0;background:#10151c;color:#dfe6ee;font:14px/1.4 system-ui}
#ui{position:fixed;top:8px;left:8px;background:#1b222cf2;padding:10px 14px;border-radius:10px;z-index:5;box-shadow:0 2px 12px #0008}
#ui label{margin-right:10px;cursor:pointer} .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:3px}
select{background:#12181f;color:#eee;border:1px solid #445;border-radius:4px;padding:2px}
#info{position:fixed;bottom:8px;left:8px;background:#1b222cd8;padding:6px 10px;border-radius:8px;font-size:12px}
</style></head><body><div id="ui">
<b>树干 / 树枝 / 树叶 分离 — PTv3</b><br>
<select id="tree"></select><br>
<label><input type="checkbox" id="c0" checked><span class="sw" style="background:#2ecc71"></span>leaf 树叶</label>
<label><input type="checkbox" id="c1" checked><span class="sw" style="background:#3478f6"></span>branch 树枝</label>
<label><input type="checkbox" id="c2" checked><span class="sw" style="background:#e74c3c"></span>trunk 树干</label>
<br><label><input type="checkbox" id="rot" checked>auto-rotate</label>
</div><div id="info"></div><script>__THREE__</script><script>
const DATA=__DATA__;
let scene,cam,ren,pts,cur=null;
function init(){scene=new THREE.Scene();scene.background=new THREE.Color(0x10151c);
 cam=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,0.01,500);cam.position.set(0,-14,4);
 ren=new THREE.WebGLRenderer({antialias:true});ren.setSize(innerWidth,innerHeight);document.body.appendChild(ren.domElement);
 const sel=document.getElementById('tree');DATA.forEach(t=>{const o=document.createElement('option');o.value=t.id;o.textContent='tree '+t.id;sel.appendChild(o)});
 sel.onchange=load;['c0','c1','c2'].forEach(id=>document.getElementById(id).onchange=load);
 load();animate();}
function load(){if(pts){scene.remove(pts);pts.geometry.dispose();pts.material.dispose()}
 const t=DATA.find(d=>d.id==document.getElementById('tree').value);const buf=new Uint8Array(atob(t.data).split('').map(ch=>ch.charCodeAt(0)));
 const N=buf.length/6;const pos=new Float32Array(buf.buffer,0,N*3).slice();const col=new Uint8Array(buf.buffer,N*3,N*3);
 const show=[c0.checked,c1.checked,c2.checked];
 const np=[];
 for(let i=0;i<N;i++){if(!show[buf[N*3+i]])continue;np.push(i);}
 const p2=new Float32Array(np.length*3),c2=new Uint8Array(np.length*3);
 np.forEach((k,j)=>{for(let a=0;a<3;a++){p2[j*3+a]=pos[k*3+a];c2[j*3+a]=col[k*3+a]}});
 const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(p2,3));
 g.setAttribute('color',new THREE.BufferAttribute(c2,3,true));
 pts=new THREE.Points(g,new THREE.PointsMaterial({size:0.02,vertexColors:true}));
 scene.add(pts);
 const b=new THREE.Box3().setFromObject(pts);const c=b.getCenter(new THREE.Vector3());pts.position.sub(c);
 document.getElementById('info').textContent=`tree ${t.id}: ${t.n} pts | leaf ${(t.counts[0]/t.n*100).toFixed(1)}% | branch ${(t.counts[1]/t.n*100).toFixed(1)}% | trunk ${(t.counts[2]/t.n*100).toFixed(1)}%`;
 cam.position.set(0,-b.getSize(new THREE.Vector3()).length()*1.2,0);}
let drag=0,px=0,py=0,down=0;
addEventListener('mousedown',e=>{drag=1;px=e.clientX;py=e.clientY});
addEventListener('mouseup',()=>drag=0);
addEventListener('mousemove',e=>{if(!drag)return;const dx=e.clientX-px,dy=e.clientY-py;px=e.clientX;py=e.clientY;
 const s=new THREE.Quaternion().setFromEuler(new THREE.Euler(dy*0.005,dx*0.005,0));pts.quaternion.premultiply(s)});
addEventListener('wheel',e=>{cam.position.multiplyScalar(1+e.deltaY*0.001)});
function animate(){requestAnimationFrame(animate);if(rot.checked&&drag==0)pts.rotation.y+=0.003;ren.render(scene,cam)}
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();ren.setSize(innerWidth,innerHeight)});
init();</script></body></html>"""

if __name__ == "__main__":
    main(*sys.argv[1:])
