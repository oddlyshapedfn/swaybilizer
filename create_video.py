import ultralytics as U
import torch
import cv2
import kornia.geometry as kgeom

def get_displacements(preds, thresh=0.5, device='cuda'):
    pts = preds[0].keypoints.data.clone()
    dims = preds[0].orig_shape
    imcenter = torch.tensor([dims[1] // 2, dims[0] // 2], device=device)

    # Loop over first tensor index which indexes multiple targets
    out = []
    pts[pts[:, :, 2] < thresh] = float('nan')
    for det in pts:
        center = det[:3, :-1].nanmean(dim=0)
        out.append(center)
    stacked = torch.stack(out)
    return stacked, stacked - imcenter

def shift(img, displacements, device='cpu'):
    outs = []
    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device)
    displacements = -1 * displacements.to(device)
    for d in displacements:
        m = torch.eye(3, device=device)[None]
        m[:, 0, -1] = d[0]
        m[:, 1, -1] = d[1]
        shifted = kgeom.warp_perspective(img, m, img.shape[-2:], align_corners=True)
        outs.append(shifted)
    return torch.stack(outs).squeeze(1)

model = U.YOLO('yolov8s-pose.pt')

boxsz = 512
fname = 'out.mp4' # (752, 1408)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(fname, fourcc, 30, (1408, 752))

reader = cv2.VideoCapture('vid.m4v')
while writer.isOpened():
    ret, frame = reader.read()
    if frame is None:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    preds = model(frame)[0]
    if preds.keypoints.data.shape[1] == 0:
        continue
    v, d = get_displacements(preds)
    shifted = shift(frame, d)
    frame = shifted[0].permute(1,2,0).to(torch.uint8).cpu().numpy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    writer.write(frame)

writer.release()
reader.release()