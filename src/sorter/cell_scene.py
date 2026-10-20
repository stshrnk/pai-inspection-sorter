"""Procedural inspection fixture. NVIDIA robot assets are referenced, never bundled."""
from __future__ import annotations
import os
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade, Sdf
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.api.materials import PhysicsMaterial, PreviewSurface
from isaacsim.core.utils.stage import get_current_stage, add_reference_to_stage
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.sensors.camera import Camera

SOURCE = np.array([0.40, 0.55, 0.045])
PART_SIZE = np.array([0.10, 0.0515, 0.06])
PART_MASS = .20
FLOOR_Z = 0.015
TRAYS = {'good': np.array([-0.40, 0.55, FLOOR_Z]),
         'defect': np.array([-0.40, 0.22, FLOOR_Z]),
         'review': np.array([0.40, 0.22, FLOOR_Z])}
TRAY_WIDTH = 0.23
ASSET_ROOT = 'https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0'
ROBOT_ASSET = '/Isaac/Samples/Rigging/Manipulator/configure_manipulator/ur10e/ur/ur_gripper.usd'


def _box(world, path, position, scale, color):
    return world.scene.add(FixedCuboid(prim_path=path, name=path.rsplit('/',1)[-1],
        position=np.array(position), scale=np.array(scale), size=1., color=np.array(color)))


def configure_gripper_physics(stage, robot_path='/ur', damping=2., max_force=8.,
                              mimic_frequency=None, mimic_damping=None):
    """Apply bounded, damped grasp dynamics as local session overrides.

    Keep native mimic relationships unless explicitly overridden for an
    experiment. GRIPPER_KEEP_NATIVE=1 preserves every asset dynamics setting.
    NVIDIA source assets stay intact.
    """
    if os.environ.get('GRIPPER_KEEP_NATIVE','0')=='1':
        print('[gripper-physics] preserving native asset dynamics',flush=True)
        return
    damping=float(os.environ.get('GRIPPER_DAMPING',damping))
    max_force=float(os.environ.get('GRIPPER_MAX_FORCE',max_force))
    if os.environ.get('GRIPPER_MIMIC_FREQUENCY'):
        mimic_frequency=float(os.environ['GRIPPER_MIMIC_FREQUENCY'])
    if os.environ.get('GRIPPER_MIMIC_DAMPING'):
        mimic_damping=float(os.environ['GRIPPER_MIMIC_DAMPING'])
    drives=[]; mimics=[]
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path)):
        if prim.GetName()=='finger_joint' and prim.HasAttribute('drive:angular:physics:stiffness'):
            prim.GetAttribute('drive:angular:physics:damping').Set(float(damping))
            prim.GetAttribute('drive:angular:physics:maxForce').Set(float(max_force))
            drives.append({'path':str(prim.GetPath()),
                'stiffness':prim.GetAttribute('drive:angular:physics:stiffness').Get(),
                'damping':float(damping),'max_force':float(max_force)})
        for attr in prim.GetAttributes():
            name=attr.GetName()
            if name.startswith('physxMimicJoint:') and name.endswith(':naturalFrequency'):
                if mimic_frequency is not None: attr.Set(float(mimic_frequency))
                mimics.append(str(prim.GetPath()))
            elif name.startswith('physxMimicJoint:') and name.endswith(':dampingRatio'):
                if mimic_damping is not None: attr.Set(float(mimic_damping))
    if not drives:
        raise RuntimeError('No driven finger_joint found in robot asset')
    print('[gripper-physics] drives='+str(drives)+' mimic_joints='+str(len(mimics))+
          ' mimic_frequency='+str(mimic_frequency if mimic_frequency is not None else 'native')+
          ' damping_ratio='+str(mimic_damping if mimic_damping is not None else 'native'),flush=True)


def create_scene(world, include_robot=True, robot_profile="robotiq85"):
    stage = get_current_stage()
    # Primitive ground avoids the default ground material's remote root discovery.
    world.scene.add_ground_plane(z_position=0., name='floor')
    _box(world, '/World/fixture', [*SOURCE[:2], FLOOR_Z/2], [.20,.15,FLOOR_Z], [.17,.20,.24])
    colors={'good': [.13,.48,.25], 'defect':[.55,.18,.14], 'review':[.65,.46,.08]}
    wall=.01
    for name,center in TRAYS.items():
        x,y,z=center
        w=TRAY_WIDTH
        _box(world, f'/World/{name}_floor', [x,y,z/2], [w+2*wall,w+2*wall,z], colors[name])
        for k,(dx,dy,sx,sy) in enumerate([(w/2+wall/2,0,wall,w+2*wall),(-w/2-wall/2,0,wall,w+2*wall),(0,w/2+wall/2,w,wall),(0,-w/2-wall/2,w,wall)]):
            _box(world, f'/World/{name}_wall{k}', [x+dx,y+dy,z+.02], [sx,sy,.04], colors[name])
    friction = PhysicsMaterial('/World/part_friction', static_friction=1.1, dynamic_friction=0.9, restitution=0.)
    part=world.scene.add(DynamicCuboid('/World/part', name='part', position=SOURCE.copy(),
        size=1., scale=PART_SIZE, mass=PART_MASS, color=np.array([.63,.66,.70]), physics_material=friction))
    # VisualCuboid binds its material stronger than descendants by default.
    # Permit the child stain to override it with its own dark surface.
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath('/World/part')).GetDirectBindingRel().SetMetadata(
        'bindMaterialAs',UsdShade.Tokens.weakerThanDescendants)
    # A mesh top face shares the rigid body's transform. Stain is visual only.
    patch=UsdGeom.Mesh.Define(stage, '/World/part/stain')
    patch.CreatePointsAttr([Gf.Vec3f(-.1,-.1,0.),Gf.Vec3f(.1,-.1,0.),Gf.Vec3f(.1,.1,0.),Gf.Vec3f(-.1,.1,0.)])
    patch.CreateFaceVertexCountsAttr([4]); patch.CreateFaceVertexIndicesAttr([0,1,2,3])
    patch.CreateSubdivisionSchemeAttr('none'); patch.CreateDoubleSidedAttr(True)
    patch.CreateDisplayColorAttr([Gf.Vec3f(.08,.055,.035)])
    stain_material=PreviewSurface('/World/stain_material',color=np.array([.08,.055,.035]),roughness=1.,metallic=0.)
    # A bound parent material overrides displayColor. Bind the stain's own
    # diffuse material so a defect changes the rendered image, not only USD.
    UsdShade.MaterialBindingAPI.Apply(patch.GetPrim()).Bind(stain_material.material)
    dome=UsdLux.DomeLight.Define(stage, '/World/lighting')
    dome.CreateIntensityAttr(65.)
    key=UsdLux.DistantLight.Define(stage, '/World/key')
    key.CreateIntensityAttr(180.); key.CreateAngleAttr(12.)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-35,20,25))
    camera=Camera('/World/inspection_camera', name='inspection_camera', resolution=(192,192), frequency=10)
    camera.set_world_pose(position=np.array([*SOURCE[:2], .80]), orientation=np.array([1.,0,0,0]), camera_axes='usd')
    # ~0.143m wide at the top face: workpiece fills the central image.
    camera.set_focal_length(.050); camera.set_horizontal_aperture(.010)
    # USD's default near clip is 1 metre, beyond this 0.8m camera's fixture.
    camera.set_clipping_range(.01,10.)
    camera.set_lens_aperture(0.)
    robot=None
    if include_robot:
        asset_root=os.environ.get('ISAAC_ASSET_ROOT',ASSET_ROOT)
        print('[startup] robot reference begin profile='+robot_profile,flush=True)
        if robot_profile=='robotiq85':
            from .robot_profiles import create_robot
            robot=create_robot(world,asset_root)
        elif robot_profile=='legacy140':
            asset=os.environ.get('ROBOT_USD',asset_root.rstrip('/')+ROBOT_ASSET)
            add_reference_to_stage(asset,'/ur')
            gripper=ParallelGripper(end_effector_prim_path='/ur/ee_link/robotiq_arg2f_base_link',
                joint_prim_names=['finger_joint'],joint_opened_positions=np.array([0.]),
                joint_closed_positions=np.array([.70]),action_deltas=None,use_mimic_joints=True)
            robot=world.scene.add(SingleManipulator(prim_path='/ur',name='robot',
                end_effector_prim_path='/ur/ee_link/robotiq_arg2f_base_link',gripper=gripper))
            robot.sorter_profile='legacy140'
        else: raise ValueError('Unknown robot profile')
        print('[startup] robot reference ready',flush=True)
        configure_gripper_physics(stage)
    return {'part':part,'camera':camera,'robot':robot,'patch':patch,'dome':dome,'key':key,
            'stain_material':stain_material}


def set_appearance(scene, condition, defective):
    """Only the generator/evaluation owns truth; classifier receives pixels."""
    stage=get_current_stage()
    gray=float(condition.get('gray',.65))
    body=stage.GetPrimAtPath('/World/part')
    # DynamicCuboid may bind a material, update its shader rather than metadata.
    material=scene['part'].get_applied_visual_material()
    if material is not None: material.set_color(np.array([gray*.96,gray,gray*1.04]))
    scene['dome'].GetIntensityAttr().Set(float(condition.get('dome',65.)))
    scene['key'].GetIntensityAttr().Set(float(condition.get('key',180.)))
    patch=scene['patch']
    if defective:
        cx=float(condition.get('patch_x',0)); cy=float(condition.get('patch_y',0))
        sx=float(condition.get('patch_w',.032)); sy=float(condition.get('patch_h',.021))
        # Child coordinates are scaled by the parent's PART_SIZE.
        z=.505
        patch.GetPointsAttr().Set([Gf.Vec3f((cx-sx/2)/PART_SIZE[0],(cy-sy/2)/PART_SIZE[1],z),
            Gf.Vec3f((cx+sx/2)/PART_SIZE[0],(cy-sy/2)/PART_SIZE[1],z),
            Gf.Vec3f((cx+sx/2)/PART_SIZE[0],(cy+sy/2)/PART_SIZE[1],z),
            Gf.Vec3f((cx-sx/2)/PART_SIZE[0],(cy+sy/2)/PART_SIZE[1],z)])
        c=float(condition.get('patch_gray',.06))
        patch.GetDisplayColorAttr().Set([Gf.Vec3f(c,c*.75,c*.55)])
        scene['stain_material'].set_color(np.array([c,c*.75,c*.55]))
        patch.MakeVisible()
    else:
        # Hide the visual patch inside the opaque body as well as changing
        # visibility. This prevents a stale shadow of a previously visible
        # stain from remaining on a good part in the realtime renderer.
        patch.GetPointsAttr().Set([Gf.Vec3f(-.1,-.1,0.),Gf.Vec3f(.1,-.1,0.),
            Gf.Vec3f(.1,.1,0.),Gf.Vec3f(-.1,.1,0.)])
        patch.MakeInvisible()


def capture_rgb(world, camera, frames=5, physics_step=None):
    # Keep physics at its configured dt. World.step(render=True) may perform
    # extra physics substeps to meet rendering_dt, which must not advance a
    # controller by a different number of ticks than its commands.
    previous_time=float(camera.get_current_frame().get('rendering_time',0.))
    for _ in range(max(60,int(frames))):
        if physics_step is None:
            world.step(render=False,update_fabric=True)
        else:
            physics_step()
        world.render()
    rgb=np.asarray(camera.get_rgb())
    if rgb.ndim!=3 or rgb.shape[2]<3 or not np.isfinite(rgb).all():
        raise RuntimeError('Inspection camera did not produce a usable RGB frame')
    current_time=float(camera.get_current_frame().get('rendering_time',0.))
    if current_time<=previous_time:
        raise RuntimeError('Inspection camera did not deliver a fresh RGB frame')
    return rgb[:,:,:3].astype(np.uint8,copy=True)


def initialize_camera(scene):
    camera=scene['camera']
    camera.set_frequency(-1)
    camera.set_world_pose(position=np.array([*SOURCE[:2], .80]), orientation=np.array([1.,0,0,0]), camera_axes='usd')
    camera.initialize()
    camera.set_world_pose(position=np.array([*SOURCE[:2], .80]), orientation=np.array([1.,0,0,0]), camera_axes='usd')
    camera.set_clipping_range(.01,10.)
    print('[camera] pose',camera.get_world_pose(camera_axes='usd'), 'focal',camera.get_focal_length(),'aperture',camera.get_horizontal_aperture(),'clipping',camera.get_clipping_range(),flush=True)
