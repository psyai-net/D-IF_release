import argparse
import os, sys
import numpy as np
import torch
import trimesh
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

from lib.renderer.mesh import load_fit_body
from lib.dataset.mesh_util import projection, load_calib, get_visibility


def visibility_subject(subject, dataset, save_folder, rotation, debug):
    
    gpu_id = queue.get()

    try:
        # run processing on GPU <gpu_id>
        # fixme: this is not working, only GPU-1 will be used for rendering
        device = torch.device(f"cuda:{gpu_id}")

        fit_file = f'./data/{dataset}/smplx/{subject}.pkl'
        rescale_fitted_body, _ = load_fit_body(fit_file, 180.0, smpl_type='smplx', smpl_gender='male')
        smpl_verts = torch.from_numpy(rescale_fitted_body.vertices).to(device).float()
        smpl_faces = torch.from_numpy(rescale_fitted_body.faces).to(device).long()

        for y in range(0, 360, 360 // rotation):

            calib_file = os.path.join(f'{save_folder}/{subject}/calib', f'{y:03d}.txt')
            vis_file = os.path.join(f'{save_folder}/{subject}/vis', f'{y:03d}.pt')

            os.makedirs(os.path.dirname(vis_file), exist_ok=True)

            if not os.path.exists(vis_file):

                calib = load_calib(calib_file).to(device)
                calib_verts = projection(smpl_verts, calib)
                (xy, z) = calib_verts.split([2, 1], dim=1)
                smpl_vis = get_visibility(xy, z, smpl_faces)

                if debug:
                    mesh = trimesh.Trimesh(
                        smpl_verts.cpu().numpy(), smpl_faces.cpu().numpy(), process=False
                    )
                    mesh.visual.vertex_colors = torch.tile(smpl_vis, (1, 3)).numpy()
                    mesh.export(vis_file.replace("pt", "obj"))

                torch.save(smpl_vis, vis_file)
    finally:
        queue.put(gpu_id)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '--dataset', type=str, default="thuman2", help='dataset name')
    parser.add_argument('-out_dir', '--out_dir', type=str, default="./debug", help='output dir')
    parser.add_argument('-num_views', '--num_views', type=int, default=36, help='number of views')
    parser.add_argument(
        '-debug', '--debug', action="store_true", help='debug mode, only render one subject'
    )
    args = parser.parse_args()

    print(f"Start Visibility Computing {args.dataset} with {args.num_views} views.")

    current_out_dir = f"{args.out_dir}/{args.dataset}_{args.num_views}views"
    os.makedirs(current_out_dir, exist_ok=True)
    print(f"Output dir: {current_out_dir}")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    
    NUM_GPUS = 2
    PROC_PER_GPU = mp.cpu_count() // NUM_GPUS
    queue = Queue()
    
    # initialize the queue with the GPU ids
    for gpu_ids in range(NUM_GPUS):
        for _ in range(PROC_PER_GPU):
            queue.put(gpu_ids)

    p = Pool(processes=mp.cpu_count(), maxtasksperchild=1)
    subjects = np.loadtxt(f"./data/{args.dataset}/all.txt", dtype=str)

    if args.debug:
        subjects = subjects[:2]

    for _ in tqdm(
        p.imap_unordered(
            partial(
                visibility_subject,
                dataset=args.dataset,
                save_folder=current_out_dir,
                rotation=args.num_views,
                debug=args.debug,
            ), subjects
        ),
        total=len(subjects)
    ):
        pass

    p.close()
    p.join()

    print('Finish Visibility Computing.')
