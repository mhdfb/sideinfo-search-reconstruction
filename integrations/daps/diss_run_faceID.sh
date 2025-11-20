# Set PYTHONPATH directly
export PYTHONPATH=DAPS:../../../DISS:../../../DISS/third_party/AdaFace

python diss_posterior_sample.py \
+model=ffhq256ddpm \
+sampler=edm_daps \
+data=diss \
+task=diss_inpainting \
+reward=face \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
data.start_id=0 data.end_id=3 name=diss_inpainting


python diss_posterior_sample.py \
+model=ffhq256ddpm \
+sampler=edm_daps \
+data=diss \
+task=diss_down_sampling \
+reward=face \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
data.start_id=0 data.end_id=3 name=diss_super
