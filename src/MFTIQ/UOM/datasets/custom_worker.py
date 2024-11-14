import logging
import math

import bpy
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
import numpy as np
import shutil
import pyquaternion as pyquat
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple


def sample_point_in_half_sphere_shell_reject_too_high(
    inner_radius: float,
    outer_radius: float,
    offset: float = 0.,
    max_z: float = None,
    rng: np.random.RandomState = None
    ) -> Tuple[float, float, float]:
    """Uniformly sample points that are in a given distance range from the origin
     and with z >= offset."""
    max_z = max_z if max_z is not None else (0.8 * outer_radius)
    assert max_z > offset
    while True:
        # normalize(3-dim standard normal) is distributed on the unit sphere surface

        xyz = kb.sample_point_in_half_sphere_shell(inner_radius, outer_radius, offset, rng=rng)
        if xyz[2] > max_z: # reject too high points
            continue
        return xyz

def euler_to_quat(euler_angles, **kwargs):
    """ Convert three (euler) angles around XYZ to a single quaternion."""
    q1 = pyquat.Quaternion(axis=[1., 0., 0.], angle=euler_angles[0], **kwargs)
    q2 = pyquat.Quaternion(axis=[0., 1., 0.], angle=euler_angles[1], **kwargs)
    q3 = pyquat.Quaternion(axis=[0., 0., 1.], angle=euler_angles[2], **kwargs)
    return tuple(q3 * q2 * q1)

def get_linear_lookat_motion_start_end(
    inner_radius: float = 1.0,
    outer_radius: float = 4.0,
    rng: np.random.RandomState = None,
    camera_start=None,
):
    """Sample a linear path which goes through the workspace center."""
    while True: # TODO: Find out why is there this "while". It seems that has no sense...
        # Sample a point near the workspace center that the path travels through
        camera_through = np.array(
            sample_point_in_half_sphere_shell_reject_too_high(0.0, inner_radius, 0.0, rng=rng)
        )
        while True:
          # Sample one endpoint of the trajectory
            camera_start = np.array(
                sample_point_in_half_sphere_shell_reject_too_high(0.0, outer_radius, 0.0, rng=rng)
            )
            if camera_start[-1] < inner_radius:
                break

        # Continue the trajectory beyond the point in the workspace center, so the
        # final path passes through that point.
        continuation = rng.rand(1) * 0.5
        camera_end = camera_through + continuation * (camera_through - camera_start)

        # Second point will probably be closer to the workspace center than the
        # first point.  Get extra augmentation by randomly swapping first and last.
        if rng.rand(1)[0] < 0.5:
            tmp = camera_start
            camera_start = camera_end
            camera_end = tmp
        return camera_start, camera_end


def get_linear_lookat_add_another_end(
    inner_radius: float = 1.0,
    outer_radius: float = 4.0,
    rng: np.random.RandomState = None,
    camera_start=None,
):
    """Sample a linear path which goes through the workspace center."""
    # Sample a point near the workspace center that the path travels through
    camera_through = np.array(
        sample_point_in_half_sphere_shell_reject_too_high(0.0, inner_radius, 0.0, rng=rng)
    )

    # Continue the trajectory beyond the point in the workspace center, so the
    # final path passes through that point.
    continuation = rng.rand(1) * 0.5
    camera_end = camera_through + continuation * (camera_through - camera_start)

    return camera_end


def get_linear_camera_motion_start_end(
        movement_speed: float,
        inner_radius: float = 8.,
        outer_radius: float = 12.,
        z_offset: float = 0.1,
        rng: np.random.RandomState = None,
        camera_start=None,
):
    """Sample a linear path which starts and ends within a half-sphere shell."""
    while True:
        if camera_start is None:
            camera_start = np.array(sample_point_in_half_sphere_shell_reject_too_high(inner_radius,
                                                                     outer_radius,
                                                                     z_offset, rng=rng))
        direction = rng.rand(3) - 0.5
        movement = direction / np.linalg.norm(direction) * movement_speed
        camera_end = camera_start + movement
        if (inner_radius <= np.linalg.norm(camera_end) <= outer_radius and
                camera_end[2] > z_offset):
            return camera_start, camera_end


def triangle_cycle_indexing(x, N):
    """
    x: current frame
    N: length of the pattern
    return idx of the pattern frame to simulate back and forth motion
    0 to N ... no changes (increasing indexes)
    N to 2N ... decreasing indexes
    2N to 3N ... same as 0 to N
    3N to 4N ... same as N to 2N
    ....
    """
    k = N - 1
    if (x % k == 0) or (x % k == N - 1):
        sign = 0
    elif x % (2 * k) > k:
        sign = -1
    else:
        sign = 1
    return k - np.abs((x % (2 * k)) - k), sign

def main(FLAGS):
    # --- Some configuration values
    # the region in which to place objects [(min), (max)]
    STATIC_SPAWN_REGION = [tuple(FLAGS.static_spawn_min), tuple(FLAGS.static_spawn_max)]
    DYNAMIC_SPAWN_REGION = [tuple(FLAGS.dynamic_spawn_min), tuple(FLAGS.dynamic_spawn_max)]
    VELOCITY_RANGE = [tuple(FLAGS.velocity_min), tuple(FLAGS.velocity_max)]

    # --- Common setups & resources
    scene, rng_main, output_dir, scratch_dir = kb.setup(FLAGS)
    scene.gravity = (0., 0., FLAGS.gravity_z)

    motion_blur = rng_main.uniform(FLAGS.min_motion_blur, FLAGS.max_motion_blur)
    if motion_blur > 0.0:
        logging.info(f"Using motion blur strength {motion_blur}")

    simulator = PyBullet(scene, scratch_dir)
    renderer = Blender(scene, scratch_dir, use_denoising=True, samples_per_pixel=FLAGS.samples_per_pixel,
                       motion_blur=motion_blur)
    kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
    gso = kb.AssetSource.from_manifest(FLAGS.gso_assets)
    shapenet = kb.AssetSource.from_manifest(FLAGS.shapenet_assets)
    if FLAGS.object_list_name == 'gso':
        objects_assets = gso
    elif FLAGS.object_list_name == 'shapenet':
        objects_assets = shapenet

    hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)

    # --- Populate the scene
    # background HDRI
    train_backgrounds, test_backgrounds = hdri_source.get_test_split(fraction=0.1)
    seed_background = FLAGS.seed if FLAGS.seed else rng_main.randint(0, 2147483647)
    rng_background = np.random.RandomState(seed=seed_background)
    if FLAGS.backgrounds_split == "train":
        logging.info("Choosing one of the %d training backgrounds...", len(train_backgrounds))
        hdri_id = rng_background.choice(train_backgrounds)
    else:
        logging.info("Choosing one of the %d held-out backgrounds...", len(test_backgrounds))
        hdri_id = rng_background.choice(test_backgrounds)
    background_hdri = hdri_source.create(asset_id=hdri_id)
    # assert isinstance(background_hdri, kb.Texture)
    logging.info("Using background %s", hdri_id)
    scene.metadata["background"] = hdri_id
    renderer._set_ambient_light_hdri(background_hdri.filename)

    # Dome
    dome = kubasic.create(asset_id="dome", name="dome",
                          friction=1.0,
                          restitution=0.0,
                          static=True, background=True)
    assert isinstance(dome, kb.FileBasedObject)
    scene += dome
    dome_blender = dome.linked_objects[renderer]
    texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
    texture_node.image = bpy.data.images.load(background_hdri.filename)

    # Camera
    seed_camera = FLAGS.seed if FLAGS.seed else rng_main.randint(0, 2147483647)
    rng_camera = np.random.RandomState(seed=seed_camera)
    rng_lookat = np.random.RandomState(seed=seed_camera + 1)
    rng_shake = np.random.RandomState(seed=seed_camera + 2)
    logging.info("Setting up the Camera...")
    scene.camera = kb.PerspectiveCamera(focal_length=FLAGS.focal_length, sensor_width=FLAGS.sensor_width)
    if FLAGS.camera == "fixed_random":
        scene.camera.position = sample_point_in_half_sphere_shell_reject_too_high(
            inner_radius=7., outer_radius=9., offset=0.1, rng=rng_camera)
        scene.camera.look_at((0, 0, 0))

    # elif FLAGS.camera == "linear_movement_linear_lookat":
    #     is_panning = True
    #     camera_inner_radius = 6.0 if is_panning else 8.0
    #     camera_start, camera_end = get_linear_camera_motion_start_end(
    #         movement_speed=rng_camera.uniform(low=FLAGS.min_camera_movement, high=FLAGS.max_camera_movement),
    #         rng=rng_camera,
    #     )
    #     lookat_start, lookat_end = get_linear_lookat_motion_start_end(rng=rng_lookat)
    #     # linearly interpolate the camera position between these two points
    #     # while keeping it focused on the center of the scene
    #     # we start one frame early and end one frame late to ensure that
    #     # forward and backward flow are still consistent for the last and first frames
    #     for frame in range(FLAGS.frame_start - 1, FLAGS.frame_end + 2):
    #         interp = ((frame - FLAGS.frame_start + 1) /
    #                   (FLAGS.frame_end - FLAGS.frame_start + 3))
    #         scene.camera.position = (interp * np.array(camera_start) +
    #                                  (1 - interp) * np.array(camera_end))
    #         scene.camera.look_at(
    #             interp * np.array(lookat_start)
    #             + (1 - interp) * np.array(lookat_end)
    #         )
    #         scene.camera.keyframe_insert("position", frame)
    #         scene.camera.keyframe_insert("quaternion", frame)

    elif FLAGS.camera == "linear_movement" or FLAGS.camera == "linear_movement_linear_lookat":
        is_panning = FLAGS.camera == "linear_movement_linear_lookat"
        camera_pos_list = list(get_linear_camera_motion_start_end(rng=rng_camera,
            movement_speed=rng_camera.uniform(low=FLAGS.min_camera_movement, high=FLAGS.max_camera_movement)
        ))
        camera_lookat_list = list(get_linear_lookat_motion_start_end(rng=rng_lookat))

        if FLAGS.camera_steps > 2:
            for _ in range(2,FLAGS.camera_steps):
                # create additional camera positions
                _, cam_point = get_linear_camera_motion_start_end(rng=rng_camera,
                                movement_speed=rng_camera.uniform(low=FLAGS.min_camera_movement, high=FLAGS.max_camera_movement),
                                camera_start=camera_pos_list[-1])
                camera_pos_list.append(cam_point)

                # create additional camera lookat positions
                look_point = get_linear_lookat_add_another_end(rng=rng_lookat,
                                                                  # movement_speed=rng_camera.uniform(
                                                                  # low=FLAGS.min_camera_movement,
                                                                  # high=FLAGS.max_camera_movement),
                                                                  camera_start=camera_lookat_list[-1])
                camera_lookat_list.append(look_point)

        elif FLAGS.camera_steps < 2:
            raise ValueError('number of camera steps has to be at least 2')
        # linearly interpolate the camera position between these two points
        # while keeping it focused on the center of the scene
        # we start one frame early and end one frame late to ensure that
        # forward and backward flow are still consistent for the last and first frames

        camera_keyframe_list = [FLAGS.frame_start - 1]
        for keyframe_order in range(1, FLAGS.camera_steps-1):
            frame_skip = keyframe_order * (FLAGS.frame_end - FLAGS.frame_start) / (FLAGS.camera_steps - 1)
            frame_skip = np.round(frame_skip).astype(int)
            camera_keyframe_list.append(camera_keyframe_list[0] + frame_skip)
        camera_keyframe_list.append(FLAGS.frame_end + 2)

        for camera_key_idx in range(FLAGS.camera_steps-1):
            c_frame_start = camera_keyframe_list[camera_key_idx]
            c_frame_end = camera_keyframe_list[camera_key_idx + 1]
            c_pos_start = camera_pos_list[camera_key_idx]
            c_pos_end = camera_pos_list[camera_key_idx + 1]
            c_lookat_start = camera_lookat_list[camera_key_idx]
            c_lookat_end = camera_lookat_list[camera_key_idx + 1]
            for frame in range(c_frame_start, c_frame_end):
                interp = ((frame - c_frame_start + 1) /
                          (c_frame_end - c_frame_start + 3))
                scene.camera.position = ((1 - interp) * np.array(c_pos_start) +
                                         (interp) * np.array(c_pos_end))
                if is_panning:
                    scene.camera.look_at((1 - interp) * np.array(c_lookat_start) +
                                             (interp) * np.array(c_lookat_end))
                else:
                    scene.camera.look_at((0, 0, 0))
                scene.camera.keyframe_insert("position", frame)
                scene.camera.keyframe_insert("quaternion", frame)
    elif FLAGS.camera == 'carlike_static':
        camera_start, camera_end = get_linear_camera_motion_start_end(rng=rng_camera,
                                                                      movement_speed=rng_camera.uniform(low=0., high=FLAGS.max_camera_movement)
                                                                      )
        camera_height = rng_camera.uniform(low=0., high=1.)
        camera_start[2] = camera_height
        camera_end[2] = camera_height
        scene.camera.position = camera_start
        scene.camera.look_at(camera_end)

    elif FLAGS.camera == "carlike_frontback_movement":
        camera_start, camera_end = get_linear_camera_motion_start_end(rng=rng_camera,
                                                                      movement_speed=rng_camera.uniform(low=FLAGS.min_camera_movement, high=FLAGS.max_camera_movement)
                                                                      )
        camera_height = rng_camera.uniform(low=0., high=1.)
        camera_start[2] = camera_height
        camera_end[2] = camera_height
        # linearly interpolate the camera position between these two points
        # while keeping it focused on the center of the scene
        # we start one frame early and end one frame late to ensure that
        # forward and backward flow are still consistent for the last and first frames
        for frame in range(FLAGS.frame_start - 1, FLAGS.frame_end + 2):
            interp = ((frame - FLAGS.frame_start + 1) /
                      (FLAGS.frame_end - FLAGS.frame_start + 3))
            scene.camera.position = ((1 - interp) * np.array(camera_start) +
                                     (interp) * np.array(camera_end))
            scene.camera.look_at(camera_end)
            scene.camera.keyframe_insert("position", frame)
            scene.camera.keyframe_insert("quaternion", frame)


    # ---- Camera Shake ----
    if "camshake" in FLAGS.general_scenario:
        current_camera_rotations = {}
        # read first, interpolation could change it after insertion of new keyframes
        for frame_idx in range(FLAGS.frame_start, FLAGS.frame_end + 1):
            current_camera_rotations[frame_idx] = scene.camera.get_value_at("quaternion", frame_idx)
        shake_magnitude = rng_shake.rand()
        for frame_idx in range(FLAGS.frame_start, FLAGS.frame_end + 1):
            shake = rng_shake.randn(3) * np.array([1, 1, 0.2]) * shake_magnitude # decrease in-plane rotation of camera shake
            # quat_shake = pyquat.Quaternion(degrees=shake)
            rot_shake = R.from_euler('xyz', shake, degrees=True).as_quat()
            # Note: Scipy's Rotation.as_quat() returns [x, y, z, w], but Kubric expects [w, x, y, z]
            quat_shake = kb.Quaternion(rot_shake[3], rot_shake[0], rot_shake[1], rot_shake[2])
            scene.camera.quaternion = quat_shake * current_camera_rotations[frame_idx]
            scene.camera.keyframe_insert("quaternion", frame_idx)


    # ---- Object placement ----
    train_split, test_split = objects_assets.get_test_split(fraction=0.1)
    if FLAGS.objects_split == "train":
        logging.info("Choosing one of the %d training objects...", len(train_split))
        active_split = train_split
    else:
        logging.info("Choosing one of the %d held-out objects...", len(test_split))
        active_split = test_split

    # add STATIC objects
    seed_static_objects = FLAGS.seed if FLAGS.seed else rng.randint(0, 2147483647)
    rng_static_objects = np.random.RandomState(seed=seed_static_objects)
    num_static_objects = rng_static_objects.randint(FLAGS.min_num_static_objects,
                                                    FLAGS.max_num_static_objects + 1)
    logging.info("Randomly placing %d static objects:", num_static_objects)
    scale_static_min = 0.75
    scale_static_max = 3.0 if FLAGS.object_list_name == 'shapenet' else 3.0
    for i in range(num_static_objects):
        obj = objects_assets.create(asset_id=rng_static_objects.choice(active_split))
        assert isinstance(obj, kb.FileBasedObject)
        scale = rng_static_objects.uniform(scale_static_min, scale_static_max)
        obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
        obj.mass = obj.mass if obj.mass > 0.0 else 1.0
        obj.metadata["scale"] = scale
        obj.metadata["mass"] = obj.mass
        scene += obj
        kb.move_until_no_overlap(obj, simulator, spawn_region=STATIC_SPAWN_REGION,
                                 rng=rng_static_objects)
        obj.friction = 1.0
        obj.restitution = 0.0
        obj.metadata["is_dynamic"] = False
        logging.info("    Added %s at %s", obj.asset_id, obj.position)

    logging.info(f"Running {20*FLAGS.frame_rate} frames of simulation to let static objects settle ...")
    _, _ = simulator.run(frame_start=-20*FLAGS.frame_rate, frame_end=0)

    # stop any objects that are still moving and reset friction / restitution
    for obj in scene.foreground_assets:
        if hasattr(obj, "velocity"):
            obj.velocity = (0., 0., 0.)
            obj.friction = 0.5
            obj.restitution = 0.5

    dome.friction = FLAGS.floor_friction
    dome.restitution = FLAGS.floor_restitution

    # Add DYNAMIC objects
    seed_dynamic_objects = FLAGS.seed if FLAGS.seed else rng_main.randint(0, 2147483647)
    rng_dynamic_objects = np.random.RandomState(seed=seed_dynamic_objects)
    num_dynamic_objects = rng_dynamic_objects.randint(FLAGS.min_num_dynamic_objects,
                                                      FLAGS.max_num_dynamic_objects + 1)
    scale_dymamic_min = 0.75
    scale_dynamic_max = 3.0 if FLAGS.object_list_name == 'shapenet' else 3.0
    logging.info("Randomly placing %d dynamic objects:", num_dynamic_objects)
    for i in range(num_dynamic_objects):
        obj = objects_assets.create(asset_id=rng_dynamic_objects.choice(active_split))
        assert isinstance(obj, kb.FileBasedObject)
        scale = rng_dynamic_objects.uniform(scale_dymamic_min, scale_dynamic_max)
        obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
        obj.metadata["scale"] = scale
        obj.mass = obj.mass if obj.mass > 0.0 else 1.0
        obj.metadata["mass"] = obj.mass
        scene += obj
        kb.move_until_no_overlap(obj, simulator, spawn_region=DYNAMIC_SPAWN_REGION,
                                 rng=rng_dynamic_objects)
        obj.velocity = (rng_dynamic_objects.uniform(*VELOCITY_RANGE) -
                        [obj.position[0], obj.position[1], 0])
        obj.metadata["is_dynamic"] = True
        logging.info("    Added %s at %s", obj.asset_id, obj.position)

    if FLAGS.save_state:
        logging.info("Saving the simulator state to '%s' prior to the simulation.",
                     output_dir / "scene.bullet")
        simulator.save_state(output_dir / "scene.bullet")

    # Run dynamic objects simulation
    logging.info("Running the simulation ...")
    animation, collisions = simulator.run(frame_start=0,
                                          frame_end=scene.frame_end + 1)

    # repeater
    if "forward_backward_cycle" in FLAGS.general_scenario:
        for obj in animation.keys():
            for frame_id in range(0, scene.frame_end + 1):
                frame_id_cycling, sign_cycling = triangle_cycle_indexing(frame_id, FLAGS.cycle_frames_for_objects)
                obj.position = animation[obj]["position"][frame_id_cycling]
                obj.quaternion = animation[obj]["quaternion"][frame_id_cycling]
                obj.velocity = np.array(animation[obj]["velocity"][frame_id_cycling]) * sign_cycling
                obj.angular_velocity = np.array(animation[obj]["angular_velocity"][frame_id_cycling]) * sign_cycling
                obj.keyframe_insert("position", frame_id)
                obj.keyframe_insert("quaternion", frame_id)
                obj.keyframe_insert("velocity", frame_id)
                obj.keyframe_insert("angular_velocity", frame_id)

    # breakpoint()

    # --- Rendering
    if FLAGS.save_state:
        logging.info("Saving the renderer state to '%s' ",
                     output_dir / "scene.blend")
        renderer.save_state(output_dir / "scene.blend")

    logging.info("Rendering the scene ...")
    data_stack = renderer.render()

    # --- Postprocessing
    kb.compute_visibility(data_stack["segmentation"], scene.assets)
    visible_foreground_assets = [asset for asset in scene.foreground_assets
                                 if np.max(asset.metadata["visibility"]) > 0]
    visible_foreground_assets = sorted(  # sort assets by their visibility
        visible_foreground_assets,
        key=lambda asset: np.sum(asset.metadata["visibility"]),
        reverse=True)

    data_stack["segmentation"] = kb.adjust_segmentation_idxs(
        data_stack["segmentation"],
        scene.assets,
        visible_foreground_assets)
    scene.metadata["num_instances"] = len(visible_foreground_assets)

    # Save to image files
    kb.write_image_dict(data_stack, output_dir)
    kb.post_processing.compute_bboxes(data_stack["segmentation"],
                                      visible_foreground_assets)

    # --- Metadata
    logging.info("Collecting and storing metadata for each object.")
    kb.write_json(filename=output_dir / "metadata.json", data={
        "flags": vars(FLAGS),
        "metadata": kb.get_scene_metadata(scene),
        "camera": kb.get_camera_info(scene.camera),
        "instances": kb.get_instance_info(scene, visible_foreground_assets),
    })
    kb.write_json(filename=output_dir / "events.json", data={
        "collisions": kb.process_collisions(
            collisions, scene, assets_subset=visible_foreground_assets),
    })

    kb.done()
    shutil.rmtree(scratch_dir)

if __name__ == '__main__':
    # --- CLI arguments
    parser = kb.ArgumentParser()
    parser.add_argument("--objects_split", choices=["train", "test"],
                        default="train")
    # Configuration for the objects of the scene
    parser.add_argument("--min_num_static_objects", type=int, default=10,
                        help="minimum number of static (distractor) objects")
    parser.add_argument("--max_num_static_objects", type=int, default=20,
                        help="maximum number of static (distractor) objects")
    parser.add_argument("--min_num_dynamic_objects", type=int, default=1,
                        help="minimum number of dynamic (tossed) objects")
    parser.add_argument("--max_num_dynamic_objects", type=int, default=3,
                        help="maximum number of dynamic (tossed) objects")
    # Configuration for the floor and background
    parser.add_argument("--floor_friction", type=float, default=0.3)
    parser.add_argument("--floor_restitution", type=float, default=0.5)
    parser.add_argument("--backgrounds_split", choices=["train", "test"],
                        default="train")

    parser.add_argument("--camera", choices=["fixed_random",
                                             "linear_movement",
                                             "carlike_static",
                                             "carlike_frontback_movement",
                                             "linear_movement_linear_lookat"],
                        default="fixed_random")
    parser.add_argument("--general_scenario",
                        # choices=["forward", "forward_backward_cycle"],
                        default='forward')
    parser.add_argument("--cycle_frames_for_objects", type=int, default=-1,
                        help="half-life of the cycle")
    parser.add_argument("--camera_steps", type=int, default=2,
                        help="number of steps for camera linear motion (default: 2, only start and end point)")
    parser.add_argument("--max_camera_movement", type=float, default=4.0)
    parser.add_argument("--min_camera_movement", type=float, default=0.0)
    parser.add_argument("--max_motion_blur", type=float, default=0.0)
    parser.add_argument("--min_motion_blur", type=float, default=0.0)

    # Configuration for the source of the assets
    parser.add_argument("--kubasic_assets", type=str,
                        default="gs://kubric-public/assets/KuBasic/KuBasic.json")
    parser.add_argument("--hdri_assets", type=str,
                        default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
    parser.add_argument("--gso_assets", type=str,
                        default="gs://kubric-public/assets/GSO/GSO.json")
    parser.add_argument("--shapenet_assets", type=str,
                        default="gs://kubric-unlisted/assets/ShapeNetCore.v2.json")

    parser.add_argument("--save_state", dest="save_state", action="store_true")
    parser.add_argument("--gravity_z", type=float, default=-9.81,
                        help="Gravity setting for z-axis.")

    parser.add_argument("--focal_length", type=float, default=35., help="focal length of the camera (mm)")
    parser.add_argument("--sensor_width", type=float, default=32., help="width of the camera sensor (mm)")

    parser.add_argument("--object_list_name", choices=["gso", "shapenet"],
                        default="gso")

    parser.add_argument("--samples_per_pixel", type=int, default=64,
                        help="renderer setting - samples per pixel")


    parser.add_argument('--static_spawn_min', nargs='+', type=float, default=[-7., -7., 0.])
    parser.add_argument('--static_spawn_max', nargs='+', type=float, default=[7., 7., 10.])
    parser.add_argument('--dynamic_spawn_min', nargs='+', type=float, default=[-5., -5., 1.])
    parser.add_argument('--dynamic_spawn_max', nargs='+', type=float, default=[5., 5., 5.])
    parser.add_argument('--velocity_min', nargs='+', type=float, default=[-4., -4., 0.])
    parser.add_argument('--velocity_max', nargs='+', type=float, default=[4., 4., 0.])

    parser.set_defaults(save_state=False, frame_end=24, frame_rate=12,
                        resolution=256)
    FLAGS = parser.parse_args()
    main(FLAGS)