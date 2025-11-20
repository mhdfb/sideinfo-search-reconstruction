# Set PYTHONPATH directly
export PYTHONPATH=diffusion-posterior-sampling:../../../DISS:../../../DISS/third_party/AdaFace

python3 diss_sample_conditions.py \
    --model_config=diffusion-posterior-sampling/configs/imagenet_model_config.yaml \
    --diffusion_config=diffusion-posterior-sampling/configs/diffusion_config.yaml \
    --task_config=diss_configs/text/diss_inpainting_config.yaml \
    --path=text_box \
    --metrics=psnr,lpips,ssim,clip

python3 diss_sample_conditions.py \
  --model_config=diffusion-posterior-sampling/configs/imagenet_model_config.yaml \
  --diffusion_config=diffusion-posterior-sampling/configs/diffusion_config.yaml \
  --task_config=diss_configs/text/diss_nonlinear_deblur_config.yaml \
  --path=text_nonlinear \
  --metrics=psnr,lpips,ssim,clip

python3 diss_sample_conditions.py \
    --model_config=diffusion-posterior-sampling/configs/imagenet_model_config.yaml \
    --diffusion_config=diffusion-posterior-sampling/configs/diffusion_config.yaml \
    --task_config=diss_configs/text/diss_super_resolution_config.yaml \
    --path=text_super \
    --metrics=psnr,lpips,ssim,clip

python3 diss_sample_conditions.py \
    --model_config=diffusion-posterior-sampling/configs/imagenet_model_config.yaml \
    --diffusion_config=diffusion-posterior-sampling/configs/diffusion_config.yaml \
    --task_config=diss_configs/text/diss_motion_deblur_config.yaml \
    --path=text_motion \
    --metrics=psnr,lpips,ssim,clip

python3 diss_sample_conditions.py \
    --model_config=diffusion-posterior-sampling/configs/imagenet_model_config.yaml \
    --diffusion_config=diffusion-posterior-sampling/configs/diffusion_config.yaml \
    --task_config=diss_configs/text/diss_gaussian_deblur_config.yaml \
    --path=text_gaussian \
    --metrics=psnr,lpips,ssim,clip
