import os
import shutil
import glob


basicPath = '/home/lisiqi/data/DeOccDepthEstimation(DODE)/'

for i in range(1, 121):
    f1 = os.path.join(basicPath, "%04d"%i)
    f2 = os.listdir(f1)
    for f in f2:
        if f == 'check':
            continue
        print(i, f)

        src = os.path.join(f1, f)
        dst = os.path.join('/lisiqi/EOccSR/', "%04d"%i, f)
        os.makedirs(dst, exist_ok=True)

        ev = os.path.join(src, 'events.npy')
        shutil.copy(ev, ev.replace(src, dst))

        imgs = glob.glob(os.path.join(src, 'frame_*.jpg'))
        for img in imgs:
            shutil.copy(img, img.replace(src, dst))

        ts = os.path.join(src, 'ts.npy')
        shutil.copy(ts, ts.replace(src, dst))

        gt = os.path.join(src, 'gt_kinect_x4_cvt.png')
        shutil.copy(gt, gt.replace(src, dst))

        flow = os.path.join(src, 'data', 'flow.npy')
        shutil.copy(flow, os.path.join(dst, 'flow.npy'))