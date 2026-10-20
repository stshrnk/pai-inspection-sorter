"""Render paired good/defect images with disjoint generation-condition groups."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from isaacsim import SimulationApp
p=argparse.ArgumentParser()
p.add_argument('--headless',action='store_true')
p.add_argument('--out-dir',default='data/inspection')
p.add_argument('--groups',type=int,default=150)
p.add_argument('--seed',type=int,default=42)
a=p.parse_args()
app=SimulationApp({'headless':a.headless,'width':640,'height':480,'renderer':'RaytracedLighting','anti_aliasing':0})
import numpy as np
from PIL import Image
from isaacsim.core.api import World
from src.sorter.cell_scene import create_scene,set_appearance,capture_rgb,initialize_camera,SOURCE

def main():
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    world=World(stage_units_in_meters=1.,physics_dt=1/200,rendering_dt=1/10)
    scene=create_scene(world,include_robot=False)
    world.reset(); initialize_camera(scene)
    rng=np.random.default_rng(a.seed)
    order=rng.permutation(a.groups)
    samples=[]
    for i,group_number in enumerate(order):
        split='train' if i<int(a.groups*.60) else 'val' if i<int(a.groups*.80) else 'test'
        group=f'condition-{group_number:03d}'
        stress=(split=='test' and i%2==0)
        condition={'gray':float(rng.uniform(.40,.82)), 'dome':float(rng.uniform(25.,110.)),
          'key':float(rng.uniform(40.,260.)), 'patch_x':float(rng.uniform(-.022,.022)),
          'patch_y':float(rng.uniform(-.010,.010)), 'patch_w':float(rng.uniform(.014,.042)),
          'patch_h':float(rng.uniform(.010,.025)), 'patch_gray':float(rng.uniform(.025,.16)),
          'stress':stress}
        if stress:
            condition['dome']=float(rng.uniform(12.,18.)); condition['key']=float(rng.uniform(15.,35.))
        for label in rng.permutation(['good','defect']):
            scene['part'].set_world_pose(position=SOURCE.copy(),orientation=np.array([1.,0,0,0]))
            scene['part'].set_linear_velocity(np.zeros(3)); scene['part'].set_angular_velocity(np.zeros(3))
            set_appearance(scene,condition,label=='defect')
            rgb=capture_rgb(world,scene['camera'],frames=6)
            relative=Path('images')/split/f'{group}_{label}.png'
            (out/relative).parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(rgb).save(out/relative)
            samples.append({'image':relative.as_posix(),'label':label,'split':split,'group':group,'condition':condition})
        if i%10==0: print(f'[dataset] groups {i+1}/{a.groups}',flush=True)
        (out/'manifest.json').write_text(json.dumps({'version':1,'seed':a.seed,'renderer':'Isaac Sim camera RGB','samples':samples},indent=2))
    print(f'[dataset] complete: {len(samples)} images',flush=True)
exit_code=1
try:
    main()
    exit_code=0
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush()
finally:
    app.close()
sys.exit(exit_code)
