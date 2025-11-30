from pathlib import Path

import builtins
import dill
import torch
import torch.nn as nn
import torch.serialization as tser
import argparse
import os
import cv2
import numpy as np
from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import Renderer, cam_crop_to_full
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv, Concat
from ultralytics.nn.modules.block import SPPF, DFL, Bottleneck, C2f
from ultralytics.nn.modules.head import Detect, Pose
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils.tal import TaskAlignedAssigner
from ultralytics.utils.loss import KeypointLoss, v8PoseLoss, BboxLoss
from ultralytics.utils import IterableSimpleNamespace


# not sure if this better or worse
torch.backends.cudnn.benchmark = True      # optimize convs for your input sizes
torch.set_grad_enabled(False)             # global inference-only mode
import cv2
cv2.setNumThreads(1)

tser.add_safe_globals([
    Conv, Concat, SPPF, DFL, Bottleneck, C2f, Detect, Pose, PoseModel,
    TaskAlignedAssigner, KeypointLoss, v8PoseLoss, BboxLoss,
    IterableSimpleNamespace, nn.Sequential, nn.Conv2d, nn.BCEWithLogitsLoss,
    nn.Upsample, nn.BatchNorm2d, nn.MaxPool2d, nn.ModuleList, nn.SiLU,
    builtins.getattr, dill._dill._load_type,
])

LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)


def main():
    parser = argparse.ArgumentParser(description='WiLoR video demo')
    parser.add_argument('--img_folder', type=str, default='demo_vid',
                        help='Folder with input videos')
    parser.add_argument('--out_folder', type=str, default='demo_vid_out',
                        help='Output folder for annotated videos')
    parser.add_argument('--save_mesh', action='store_true', default=False)
    parser.add_argument('--rescale_factor', type=float, default=2.0)
    parser.add_argument('--file_type', nargs='+',
                        default=['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4'])
    args = parser.parse_args()

    print("[INFO] Loading model and detector...")
    model, model_cfg = load_wilor(
        checkpoint_path='./pretrained_models/wilor_final.ckpt',
        cfg_path='./pretrained_models/model_config.yaml'
    )
    detector = YOLO('./pretrained_models/detector.pt')
    detector.fuse()  # fuse Conv+BN for faster inference not sure if good 
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    model = model.to(device).eval()
    detector = detector.to(device)

    os.makedirs(args.out_folder, exist_ok=True)

    # Collect video paths
    video_paths = [v for ext in args.file_type for v in Path(args.img_folder).glob(ext)]
    if not video_paths:
        print(f"[WARN] No videos found in {args.img_folder}")
        return

    print(f"[INFO] Found {len(video_paths)} videos to process.\n")

    for vid_i, video_path in enumerate(video_paths, 1):
        video_path = str(video_path)
        video_name = Path(video_path).stem
        print(f"[{vid_i}/{len(video_paths)}] Processing video: {video_name}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = os.path.join(args.out_folder, f'{video_name}_annotated.mp4')
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        frame_idx = 0
        while True:
            ret, img_cv2 = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % 50 == 0 or frame_idx == 1:
                print(f"  - Frame {frame_idx}/{total_frames}")

            detections = detector(img_cv2, conf=0.6, verbose=False)[0] # this is messing up
            bboxes, is_right = [], []
            for det in detections:
                Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
                is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
                bboxes.append(Bbox[:4].tolist())

            if not bboxes:
                writer.write(img_cv2)
                continue

            boxes = np.stack(bboxes)
            right = np.stack(is_right)
            dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right,
                                    rescale_factor=args.rescale_factor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)

            all_verts, all_cam_t, all_right = [], [], []

            for batch in dataloader:
                batch = recursive_to(batch, device)
                # with torch.no_grad():
                out = model(batch)
                # print(f"    [Model] Batch processed, got {len(batch['img'])} samples.")

                multiplier = (2 * batch['right'] - 1)
                pred_cam = out['pred_cam']
                pred_cam[:, 1] = multiplier * pred_cam[:, 1]
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal_length = (
                    model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                )
                pred_cam_t_full = cam_crop_to_full(
                    pred_cam, box_center, box_size, img_size, scaled_focal_length
                ).detach().cpu().numpy()

                for n in range(len(batch['img'])):
                    verts = out['pred_vertices'][n].detach().cpu().numpy()
                    cam_t = pred_cam_t_full[n]
                    is_r = batch['right'][n].cpu().numpy()
                    verts[:, 0] = (2 * is_r - 1) * verts[:, 0]
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
                    all_right.append(is_r)

                    if args.save_mesh:
                        renderer.vertices_to_trimesh(verts, cam_t, LIGHT_PURPLE, is_right=is_r) \
                            .export(os.path.join(args.out_folder, f'{video_name}_f{frame_idx}_{n}.obj'))

            if all_verts:
                cam_view = renderer.render_rgba_multiple(
                    all_verts, cam_t=all_cam_t,
                    render_res=img_size[0], is_right=all_right,
                    mesh_base_color=LIGHT_PURPLE, scene_bg_color=(1, 1, 1),
                    focal_length=float(scaled_focal_length)
                )
                input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
                input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
                blended = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
                frame_out = (255 * blended[:, :, ::-1]).astype(np.uint8)
            else:
                frame_out = img_cv2

            writer.write(frame_out)
            if frame_idx % 500 == 0:
                print(f"    [Render] Saved frame {frame_idx}/{total_frames}")

        cap.release()
        writer.release()
        print(f"[DONE] Saved annotated video → {out_path}\n")


def project_full_img(points, cam_trans, focal_length, img_res):
    img_w, img_h = float(img_res[0]), float(img_res[1])
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = K[1, 1] = focal_length
    K[0, 2], K[1, 2] = img_w / 2.0, img_h / 2.0
    points = points + cam_trans
    points = points / points[..., -1:]
    return (K @ points.T).T[..., :-1]


if __name__ == '__main__':
    main()
