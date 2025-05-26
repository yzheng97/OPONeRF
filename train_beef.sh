# conda environment neuray
# # trainning
CUDA_VISIBLE_DEVICES=6 python run_training.py \
    --cfg configs/train/gen/neuray_gen_depth_train_beef.yaml

# # validation 
# CUDA_VISIBLE_DEVICES=5 python render.py \
#     --cfg=configs/gen/neuray_gen_depth_beef.yaml \
#     --database_name=example/beef_0 --pose_type=eval \
#     --train_views_for_valid 5.5
# CUDA_VISIBLE_DEVICES=5 python eval.py \
#     --dir_pr=data/render/example/beef_0/neuray_gen_depth_train_beef-eval/ \
#     --dir_gt=data/render/example/beef_0/gt/

# # # test in new frame
# for i in {1..21..1}
# do
#     CUDA_VISIBLE_DEVICES=0 python render.py \
#         --cfg=configs/gen/neuray_gen_depth_beef.yaml \
#         --database_name=example/beef_$i --pose_type=eval 
#     CUDA_VISIBLE_DEVICES=0 python eval.py \
#     --dir_pr=data/render/example/beef_$i/neuray_gen_depth_train_beef-eval/ \
#     --dir_gt=data/render/example/beef_$i/gt/
# done