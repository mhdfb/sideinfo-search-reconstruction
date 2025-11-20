# Set PYTHONPATH directly
export PYTHONPATH=blind-dps:../../../DISS:../../../DISS/third_party/AdaFace

python3 diss_deblur.py \
    --img_model_config=blind-dps/configs/model_config.yaml \
    --kernel_model_config=blind-dps/configs/kernel_model_config.yaml \
    --diffusion_config=diss_configs/diss_diffusion_config.yaml \
    --task_config=diss_configs/faceID/diss_motion_deblur.yaml \
    --reg_ord=1 \
    --reg_scale=1.0 \
    --path=diss_faceID \
    --metrics=psnr,lpips,ssim,adaface


python3 diss_deblur.py \
    --img_model_config=diss_configs/text/imagenet_model_config.yaml \
    --kernel_model_config=blind-dps/configs/kernel_model_config.yaml \
    --diffusion_config=diss_configs/diss_diffusion_config.yaml \
    --task_config=diss_configs/text/diss_motion_deblur.yaml \
    --reg_ord=1 \
    --reg_scale=1.0 \
    --path=diss_text \
    --metrics=psnr,lpips,ssim,clip
