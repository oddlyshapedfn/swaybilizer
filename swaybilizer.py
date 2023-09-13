import ultralytics as U
import torch
import cv2
import kornia.geometry as kgeom

IN_NAME='tdw.m4v'
OUT_NAME='out.mp4'
WIDTH=1280
HEIGHT=720
FPS=30
DECIMATE=1
DEVICE='cuda'

# MODEL = 'yolov8n-pose.pt' # Faster but less accurate
MODEL = 'yolov8s-pose.pt'

def translation_matrix(dx, dy, device='cpu'):
    return torch.tensor(
        [
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1]
        ],
        device=device
    ).float()

def get_transforms(preds, thresh=0.5, device='cpu'):
    pts = preds[0].keypoints.data.clone()
    dims = preds[0].orig_shape
    imcenter = torch.tensor([dims[1] // 2, dims[0] // 2], device=device)

    # Loop over first tensor index which indexes multiple targets
    out = []
    pts[pts[:, :, 2] < thresh] = float('nan')
    for det in pts:
        coords = det[..., :-1]
        center = -1 * (coords[:3].nanmean(dim=0) - imcenter)

        # Calculate the angle from the line segment connecting the eyes
        slope = coords[1] - coords[2]
        angle = -1* torch.atan2(slope[1], slope[0])
        print(angle*57.3)

        m0 = torch.tensor(
            [
                [1, 0, center[0]],
                [0, 1, center[1]],
                [0, 0, 1]
            ],
            device=device
        )
        m1 = torch.tensor(
            [
                [torch.cos(angle), -1 * torch.sin(angle), 0],
                [torch.sin(angle),      torch.cos(angle), 0],
                [0, 0, 1]
            ],
            device=device
        )

        t0 = translation_matrix(-1 * dims[1] // 2, -1 * dims[0] // 2, device=device)
        t1 = translation_matrix(dims[1] // 2, dims[0] // 2, device=device)

        # The transforms we need to perform are...
        # 1. Move the stabilized point to the center of the frame
        # 2. Move the center of the frame to origin (top left corner)
        # 3. Rotate the image around the origin so that the eye points are level
        # 4. Undo 2) by moving the image back to the center.
        # The standard 3x3 transform matrix rotates about the origin, so to rotate around
        # The frame center, we need to move the image there and back (virtually)
        transform = t1.matmul(m1.matmul(t0.matmul(m0)))

        out.append(transform[None])

    stacked = torch.stack(out)
    return stacked

def shift(img, transforms, device='cpu'):
    outs = []
    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device)

    transforms = transforms.to(device)
    for d in transforms:
        shifted = kgeom.warp_perspective(img, d, img.shape[-2:], align_corners=True)
        outs.append(shifted)
    return torch.stack(outs).squeeze(1)

if __name__ == '__main__':
    model = U.YOLO(MODEL)

    fname = OUT_NAME
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(fname, fourcc, FPS, (WIDTH, HEIGHT))

    frame_count = 0
    reader = cv2.VideoCapture(IN_NAME)
    while writer.isOpened():
        print(frame_count)
        frame_count += 1
        ret, frame = reader.read()
        if frame is None:
            break
        if frame_count % DECIMATE != 0:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preds = model(frame)[0]
        if preds.keypoints.data.shape[1] == 0:
            continue
        d = get_transforms(preds, device=DEVICE)
        shifted = shift(frame, d, device=DEVICE)
        frame = shifted[0].permute(1,2,0).to(torch.uint8).cpu().numpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.write(frame)

    writer.release()
    reader.release()
