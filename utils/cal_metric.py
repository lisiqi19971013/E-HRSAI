from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


def calpsnr(gt, pred):
    if len(gt.shape) == 4:
        bs = gt.shape[0]
        psnr = 0
        for b in range(bs):
            psnr += peak_signal_noise_ratio(gt[b], pred[b], data_range=gt[b].max() - gt[b].min())
        return psnr/bs
    elif len(gt.shape) == 3:
        return peak_signal_noise_ratio(gt, pred, data_range=gt.max() - gt.min())


def calssim(gt, pred):
    if len(gt.shape) == 4:
        if gt.shape[3] != 1 and gt.shape[3] != 3:
            gt = gt.transpose([0,2,3,1])
            pred = pred.transpose([0,2,3,1])
        bs = gt.shape[0]
        ssim = 0
        multichannel = gt.shape[2] == 3
        if multichannel:
            for b in range(bs):
                ssim += structural_similarity(gt[b], pred[b], data_range=gt[b].max() - gt[b].min(), multichannel=True, gaussian_weights=True)
        else:
            for b in range(bs):
                ssim += structural_similarity(gt[b, :, :, 0], pred[b, :, :, 0], data_range=gt[b].max() - gt[b].min(), multichannel=False, gaussian_weights=True)
        return ssim/bs

    elif len(gt.shape) == 3:
        # print(gt.shape)
        if gt.shape[2] != 1 and gt.shape[2] != 3:
            gt = gt.transpose([1, 2, 0])
            pred = pred.transpose([1, 2, 0])
        multichannel = gt.shape[2] == 3
        # print(gt.shape)
        if multichannel:
            return structural_similarity(gt, pred, data_range=gt.max() - gt.min(), multichannel=True, gaussian_weights=True)
        else:
            return structural_similarity(gt[:,:,0], pred[:,:,0], data_range=gt.max() - gt.min(), multichannel=False, gaussian_weights=True)
    elif len(gt.shape) == 2:
        return structural_similarity(gt, pred, data_range=gt.max() - gt.min(), multichannel=False, gaussian_weights=True)


def psnr_batch(output, gt):
    B = output.shape[0]
    output = output.permute(0, 2, 3, 1).contiguous().cpu().detach().numpy()
    output[output<0] = 0
    output[output>1] = 1
    gt = gt.permute(0, 2, 3, 1).contiguous().cpu().detach().numpy()
    gt[gt<0] = 0
    gt[gt>1] = 1
    psnr = 0
    for i in range(B):
        psnr += peak_signal_noise_ratio(gt[i], output[i])
    return psnr/B


def ssim_batch(output, gt):
    B = output.shape[0]
    output = output.permute(0, 2, 3, 1).contiguous().cpu().detach().numpy()
    output[output < 0] = 0
    output[output > 1] = 1
    gt = gt.permute(0, 2, 3, 1).contiguous().cpu().detach().numpy()
    gt[gt < 0] = 0
    gt[gt > 1] = 1
    ssim = 0
    for i in range(B):
        ssim += structural_similarity(gt[i], output[i], multichannel=True)
    return ssim/B