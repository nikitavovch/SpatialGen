export PYTHONPATH=$PYTHONPATH:$PWD
echo "PYTHONPATH: $PYTHONPATH"

export CUDA_LAUNCH_BLOCKING=1


INFER_TRAGS=(
    "Text23D_exp_spatialgen_16view_sample_0"
)

INFER_SEEDS=(2024)


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
        --scene_id scene_00000 \
        --style_prompt "A Traditional Chinese Style living room with rosewood furniture, jade ornaments, and silk screens, arranged in a feng shui layout with a central rosewood coffee table and a black lacquer sideboard. Warm, natural lighting enhances the deep red and gold accents, while paper lanterns and carved details add to the aesthetic. The color palette includes deep red, gold, black lacquer, jade green, and warm brown, creating a harmonious and elegant atmosphere." \
        --guidance_scale 3.5 \
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