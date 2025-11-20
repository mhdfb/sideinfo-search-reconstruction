# Set PYTHONPATH directly
export PYTHONPATH=diffusion-posterior-sampling:../../../DISS:../../../DISS/third_party/AdaFace

python3 diss_sample_conditions.py \
    --model_config=diffusion-posterior-sampling/configs/model_config.yaml \
    --diffusion_config=diffusion-posterior-sampling/configs/diffusion_config.yaml \
    --task_config=diss_configs/faceID/diss_inpainting_config.yaml \
    --path=face_box \
    --metrics=psnr,lpips,ssim,adaface

python3 diss_sample_conditions.py \
    --model_config=diffusion-posterior-sampling/configs/model_config.yaml \
    --diffusion_config=diffusion-posterior-sampling/configs/diffusion_config.yaml \
    --task_config=diss_configs/faceID/diss_nonlinear_deblur_config.yaml \
    --path=face_nonlinear \
    --metrics=psnr,lpips,ssim,adaface

python3 diss_sample_conditions.py \
    --model_config=diffusion-posterior-sampling/configs/model_config.yaml \
    --diffusion_config=diffusion-posterior-sampling/configs/diffusion_config.yaml \
    --task_config=diss_configs/faceID/diss_super_resolution_config.yaml \
    --path=face_super \
    --metrics=psnr,lpips,ssim,adaface

python3 diss_sample_conditions.py \
    --model_config=diffusion-posterior-sampling/configs/model_config.yaml \
    --diffusion_config=diffusion-posterior-sampling/configs/diffusion_config.yaml \
    --task_config=diss_configs/faceID/diss_motion_deblur_config.yaml \
    --path=face_motion \
    --metrics=psnr,lpips,ssim,adaface

python3 diss_sample_conditions.py \
    --model_config=diffusion-posterior-sampling/configs/model_config.yaml \
    --diffusion_config=diffusion-posterior-sampling/configs/diffusion_config.yaml \
    --task_config=diss_configs/faceID/diss_gaussian_deblur_config.yaml \
    --path=face_gaussian \
    --metrics=psnr,lpips,ssim,adaface
