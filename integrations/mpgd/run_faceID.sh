export PYTHONPATH=../../../DISS/third_party/AdaFace

# run RFJS

python run_inverse_diss.py \
        --model_config=diss_configs/model_config.yaml \
        --diffusion_config=diss_configs/mpgd_diffusion_search_config.yaml \
        --task_config=diss_configs/diss_inpainting_config.yaml \
        --timestep=100 \
        --scale=4 \
        --method="mpgd_wo_proj" \
        --save_dir='./outputs_faceID_RFJS/' \
        --n_images=3 \
        --seed=37


python run_inverse_diss.py \
        --model_config=diss_configs/model_config.yaml \
        --diffusion_config=diss_configs/mpgd_diffusion_search_config.yaml \
        --task_config=diss_configs/diss_super_resolution_config.yaml \
        --timestep=100 \
        --scale=4 \
        --method="mpgd_wo_proj" \
        --save_dir='./outputs_faceID_RFJS/' \
        --n_images=3 \
        --seed=37


python run_inverse_diss.py \
        --model_config=diss_configs/model_config.yaml \
        --diffusion_config=diss_configs/mpgd_diffusion_search_config.yaml \
        --task_config=diss_configs/diss_gaussian_deblur_config.yaml \
        --timestep=100 \
        --scale=4 \
        --method="mpgd_wo_proj" \
        --save_dir='./outputs_faceID_RFJS/' \
        --n_images=3 \
        --seed=37


# run MPGD

python run_inverse_diss.py \
        --model_config=diss_configs/model_config.yaml \
        --diffusion_config=diss_configs/mpgd_diffusion_search_config.yaml \
        --task_config=diss_configs/diss_inpainting_mpgd.yaml \
        --timestep=100 \
        --scale=4 \
        --method="mpgd_wo_proj" \
        --save_dir='./outputs_faceID_MPGD/' \
        --n_images=3 \
        --seed=37


python run_inverse_diss.py \
        --model_config=diss_configs/model_config.yaml \
        --diffusion_config=diss_configs/mpgd_diffusion_search_config.yaml \
        --task_config=diss_configs/diss_super_resolution_mpgd.yaml \
        --timestep=100 \
        --scale=4 \
        --method="mpgd_wo_proj" \
        --save_dir='./outputs_faceID_MPGD/' \
        --n_images=3 \
        --seed=37


python run_inverse_diss.py \
        --model_config=diss_configs/model_config.yaml \
        --diffusion_config=diss_configs/mpgd_diffusion_search_config.yaml \
        --task_config=diss_configs/diss_gaussian_deblur_mpgd.yaml \
        --timestep=100 \
        --scale=4 \
        --method="mpgd_wo_proj" \
        --save_dir='./outputs_faceID_MPGD/' \
        --n_images=3 \
        --seed=37
