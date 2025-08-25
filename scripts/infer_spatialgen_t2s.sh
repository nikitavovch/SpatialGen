export PYTHONPATH=$PYTHONPATH:$PWD
echo "PYTHONPATH: $PYTHONPATH"

export CUDA_LAUNCH_BLOCKING=1


INFER_TRAGS=(
    "Text23D_exp_spatialgen_16view_sample_0"
    "Text23D_exp_spatialgen_16view_sample_1"
    "Text23D_exp_spatialgen_16view_sample_2"
)
INFER_SEEDS=(2024 2025 2026)

for i in "${!INFER_TRAGS[@]}"; do
    infertag="${INFER_TRAGS[$i]}"
    inferseed="${INFER_SEEDS[$i]}"

    python3 src/inference_sd.py \
        --config_file configs/test_spatialgen_sd21.yaml \
        --tag Text23D_exp_spatialgen \
        --allow_tf32 \
        --seed ${inferseed} \
        --use_controlnet \
        --infer_tag=$infertag \
        --guidance_scale 3.5 \
        --styled_prompt_idx $i \
        --output_dir ./out \
        opt.pretrained_model_name_or_path=manycore-research/SpatialGen-1.0 \
        opt.input_res=512 \
        opt.num_input_views=1 \
        opt.num_views=16 \
        opt.prediction_type=v_prediction \
        opt.use_layout_prior=true \
        opt.use_scene_coord_map=true \
        opt.use_metric_depth=false \
        opt.input_concat_binary_mask=true \
        opt.input_concat_warpped_image=true \
        opt.spatiallm_data_dir=./none \
        opt.structured3d_data_dir=./none \
        opt.trajectory_sampler_type=spiral
done
# bash scripts/infer.sh \
#     src/inference_sd.py \
#     configs/spatialgen_sd21.yaml \
#     Text23D_exp_spatialgen_flux \
#     -seed 2025 \
#     --use_controlnet \
#     --infer_tag=Text23D_exp_spatialgen_fluxcontrolnet_16view_sd21_0823 \
#     --output_dir /data-nas/data/experiments/zhenqing/spatialgen/out  \
#     --guidance_scale 3.5 \
#     opt.pretrained_model_name_or_path=/data-nas/experiments/zhenqing/diffsplat/out/8rgbsscm_warp_512_sd21_ckpt020000  \
#     opt.input_res=512 \
#     opt.num_input_views=1 \
#     opt.num_views=16 \
#     opt.prediction_type=v_prediction \
#     opt.use_layout_prior=true \
#     opt.use_scene_coord_map=true \
#     opt.use_metric_depth=false \
#     opt.input_concat_binary_mask=true \
#     opt.input_concat_warpped_image=true \
#     opt.koolai_data_dir=/data-nas/data/dataset/qunhe/PanoRoom/roomverse_data/processed_data_spiral/ \
#     opt.train_split_file=/data-nas/data/dataset/qunhe/PanoRoom/roomverse_data/new_57k_perspective_trains.txt \
#     opt.invalid_split_file=/data-nas/data/dataset/qunhe/PanoRoom/roomverse_data/new_57k_perspective_invalid_scenes.txt \
#     opt.spatiallm_data_dir=./none \
#     opt.structured3d_data_dir=./none \
#     opt.trajectory_sampler_type=spiral