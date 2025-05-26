import numpy as np
psnr = []
ssim = []
lpips = []
psnr2 = []
ssim2 = []
lpips2 = []
cnt = []

#trimming_ori 30.36
#discussion_ori 28.25
#vr_ori  29.29 
for i in range(1,22):
    
    # with open(f'/opt/data/zy/zkf/NeuRay/data/render/example/no_noise/beef_{i}/neuray_gen_depth_train_beef_ori2-eval/eval.txt', 'r') as f:
    with open(f'/opt/yhr/scarf/OPONeRF/data/render/example/beef_{i}/neuray_gen_depth_train_beef_source-eval/eval.txt', 'r') as f:
        st = f.read().split()
        # print(float(st[1]))
        psnr.append(float(st[0]))
        ssim.append(float(st[1]))
        lpips.append(float(st[2]))
        # psnr2.append(float(st[7]))
        # ssim2.append(float(st[9]))
        # lpips2.append(float(st[11]))
        
# for i in range(1,12):
    
#     # with open(f'/opt/data/zy/zkf/NeuRay/data/render/example/no_noise/beef_{i}/neuray_gen_depth_train_beef_ori2-eval/eval.txt', 'r') as f:
#     with open(f'/opt/yhr/scarf/OPONeRF/data/render/clevr/clevr8_{i}/neuray_gen_depth_train_clevr8_mask-eval/eval.txt', 'r') as f:
#         st = f.read().split()
#         # print(float(st[1]))
#         psnr.append(float(st[0]))
#         ssim.append(float(st[1]))
#         lpips.append(float(st[2]))
        
w_str = f'psnr: {np.mean(psnr):.4f}  ssim: {np.mean(ssim):.4f}  lpips: {np.mean(lpips):.4f}  '
# w_str = f'psnr: {np.mean(psnr):.4f}  ssim: {np.mean(ssim):.4f}  lpips: {np.mean(lpips):.4f}  psnr2: {np.mean(psnr2):.4f}  ssim2: {np.mean(ssim2):.4f}  lpips2: {np.mean(lpips2):.4f} '
print(w_str)
# outputdir = './results/obla1.txt'
# with open(outputdir, 'w') as f:
#     f.write(w_str)