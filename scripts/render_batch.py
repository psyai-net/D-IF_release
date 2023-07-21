import argparse
import os, sys
import cv2
import trimesh
import numpy as np
import random
import math
import random
from tqdm import tqdm

# multi-thread
from functools import partial
from multiprocessing import Pool, Queue
import multiprocessing as mp

# to remove warning from numba
# "Numba: Attempted to fork from a non-main thread, the TBB library may be in an invalid state in the child process.""
import numba
numba.config.THREADING_LAYER = 'workqueue'

sys.path.append(os.path.join(os.getcwd()))


def render_sides(render_types, rndr, rndr_smpl, y, save_folder, subject, smpl_type, side):

    if "normal" in render_types:
        opengl_util.render_result(
            rndr, 1, os.path.join(save_folder, subject, f"normal_{side}", f'{y:03d}.png')
        )

    if "depth" in render_types:
        opengl_util.render_result(
            rndr, 2, os.path.join(save_folder, subject, f"depth_{side}", f'{y:03d}.png')
        )

    if smpl_type != "none":

        opengl_util.render_result(
            rndr_smpl, 1, os.path.join(save_folder, subject, f"T_normal_{side}", f'{y:03d}.png')
        )

        if "depth" in render_types:
            opengl_util.render_result(
                rndr_smpl, 2, os.path.join(save_folder, subject, f"T_depth_{side}", f'{y:03d}.png')
            )


def render_subject(subject, dataset, save_folder, rotation, size, render_types, egl):

    gpu_id = queue.get()

    try:
        # run processing on GPU <gpu_id>
        # fixme: this is not working, only GPU-1 will be used for rendering
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        initialize_GL_context(width=size, height=size, egl=egl)

        scale = 100.0
        up_axis = 1
        smpl_type = "smplx"

        mesh_file = os.path.join(f'./data/{dataset}/scans/{subject}', f'{subject}.obj')
        smplx_file = f'./data/{dataset}/{smpl_type}/{subject}.obj'
        tex_file = f'./data/{dataset}/scans/{subject}/material0.jpeg'
        fit_file = f'./data/{dataset}/{smpl_type}/{subject}.pkl'

        vertices, faces, normals, faces_normals, textures, face_textures = load_scan(
            mesh_file, with_normal=True, with_texture=True
        )

        # center
        scan_scale = 1.8 / (vertices.max(0)[up_axis] - vertices.min(0)[up_axis])
        rescale_fitted_body, joints = load_fit_body(
            fit_file, scale, smpl_type=smpl_type, smpl_gender='male'
        )

        os.makedirs(os.path.dirname(smplx_file), exist_ok=True)
        trimesh.Trimesh(rescale_fitted_body.vertices / scale,
                        rescale_fitted_body.faces).export(smplx_file)

        vertices *= scale
        vmin = vertices.min(0)
        vmax = vertices.max(0)
        vmed = joints[0]
        vmed[up_axis] = 0.5 * (vmax[up_axis] + vmin[up_axis])

        rndr_smpl = ColorRender(width=size, height=size, egl=egl)
        rndr_smpl.set_mesh(
            rescale_fitted_body.vertices, rescale_fitted_body.faces, rescale_fitted_body.vertices,
            rescale_fitted_body.vertex_normals
        )
        rndr_smpl.set_norm_mat(scan_scale, vmed)

        # camera
        cam = Camera(width=size, height=size)
        cam.ortho_ratio = 0.4 * (512 / size)

        prt, face_prt = prt_util.computePRT(mesh_file, scale, 10, 2)
        rndr = PRTRender(width=size, height=size, ms_rate=16, egl=egl)

        # texture
        texture_image = cv2.cvtColor(cv2.imread(tex_file), cv2.COLOR_BGR2RGB)

        tan, bitan = compute_tangent(normals)
        rndr.set_norm_mat(scan_scale, vmed)
        rndr.set_mesh(
            vertices,
            faces,
            normals,
            faces_normals,
            textures,
            face_textures,
            prt,
            face_prt,
            tan,
            bitan,
            np.zeros((vertices.shape[0], 3)),
        )
        rndr.set_albedo(texture_image)

        for y in range(0, 360, 360 // rotation):

            cam.near = -100
            cam.far = 100
            cam.sanity_check()

            R = opengl_util.make_rotate(0, math.radians(y), 0)
            R_B = opengl_util.make_rotate(0, math.radians((y + 180) % 360), 0)

            rndr.rot_matrix = R
            rndr.set_camera(cam)

            if smpl_type != "none":
                rndr_smpl.rot_matrix = R
                rndr_smpl.set_camera(cam)

            dic = {'ortho_ratio': cam.ortho_ratio, 'scale': scan_scale, 'center': vmed, 'R': R}

            if "light" in render_types:

                # random light
                shs = np.load('./scripts/env_sh.npy')
                sh_id = random.randint(0, shs.shape[0] - 1)
                sh = shs[sh_id]
                sh_angle = 0.2 * np.pi * (random.random() - 0.5)
                sh = opengl_util.rotateSH(sh, opengl_util.make_rotate(0, sh_angle, 0).T)
                dic.update({"sh": sh})

                rndr.set_sh(sh)
                rndr.analytic = False
                rndr.use_inverse_depth = False

            # ==================================================================

            # calib
            calib = opengl_util.load_calib(dic, render_size=size)

            export_calib_file = os.path.join(save_folder, subject, 'calib', f'{y:03d}.txt')
            os.makedirs(os.path.dirname(export_calib_file), exist_ok=True)
            np.savetxt(export_calib_file, calib)

            # ==================================================================

            # front render
            rndr.display()
            rndr_smpl.display()

            opengl_util.render_result(
                rndr, 0, os.path.join(save_folder, subject, 'render', f'{y:03d}.png')
            )

            render_sides(render_types, rndr, rndr_smpl, y, save_folder, subject, smpl_type, "F")
            # ==================================================================

            # back render
            cam.near = 100
            cam.far = -100
            cam.sanity_check()
            rndr.set_camera(cam)
            rndr_smpl.set_camera(cam)

            rndr.display()
            rndr_smpl.display()

            render_sides(render_types, rndr, rndr_smpl, y, save_folder, subject, smpl_type, "B")

    finally:
        queue.put(gpu_id)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '--dataset', type=str, default="thuman2", help='dataset name')
    parser.add_argument('-out_dir', '--out_dir', type=str, default="./debug", help='output dir')
    parser.add_argument('-num_views', '--num_views', type=int, default=36, help='number of views')
    parser.add_argument('-size', '--size', type=int, default=512, help='render size')
    parser.add_argument(
        '-debug', '--debug', action="store_true", help='debug mode, only render one subject'
    )
    parser.add_argument(
        '-headless', '--headless', action="store_true", help='headless rendering with EGL'
    )
    args = parser.parse_args()

    # rendering setup
    if args.headless:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    else:
        os.environ["PYOPENGL_PLATFORM"] = ""

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # shoud be put after PYOPENGL_PLATFORM
    import lib.renderer.opengl_util as opengl_util
    from lib.renderer.mesh import load_fit_body, load_scan, compute_tangent
    import lib.renderer.prt_util as prt_util
    from lib.renderer.gl.init_gl import initialize_GL_context
    from lib.renderer.gl.prt_render import PRTRender
    from lib.renderer.gl.color_render import ColorRender
    from lib.renderer.camera import Camera

    print(
        f"Start Rendering {args.dataset} with {args.num_views} views, {args.size}x{args.size} size."
    )

    current_out_dir = f"{args.out_dir}/{args.dataset}_{args.num_views}views"
    os.makedirs(current_out_dir, exist_ok=True)
    print(f"Output dir: {current_out_dir}")

    subjects = np.loadtxt(f"/data/yangxueting/ICON_orl/data/{args.dataset}/all.txt", dtype=str)

    if args.debug:
        subjects = subjects[:2]
        render_types = ["light", "normal", "depth"]
    else:
        random.shuffle(subjects)
        render_types = ["light", "normal"]

    print(f"Rendering types: {render_types}")

    NUM_GPUS = 2
    PROC_PER_GPU = mp.cpu_count() // NUM_GPUS

    queue = Queue()

    # initialize the queue with the GPU ids
    for gpu_ids in range(NUM_GPUS):
        for _ in range(PROC_PER_GPU):
            queue.put(gpu_ids)

    with Pool(processes=mp.cpu_count(), maxtasksperchild=1) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                partial(
                    render_subject,
                    dataset=args.dataset,
                    save_folder=current_out_dir,
                    rotation=args.num_views,
                    size=args.size,
                    egl=args.headless,
                    render_types=render_types,
                ),
                subjects,
            ),
            total=len(subjects)
        ):
            pass

    pool.close()
    pool.join()

    print('Finish Rendering.')
