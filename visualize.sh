python scripts/visualize_hdf5_actions.py \
  --dataset_path /media/damoxing/datasets/myendless/pick_up_board_black_bg \
  --action_mode delta_joint \
  --action_target absolute \
  --num_samples 6 \
  --max_steps 500 \
  --output_dir ./logs/action_viz_joint

python scripts/visualize_hdf5_actions.py \
  --dataset_path /media/damoxing/datasets/myendless/pick_up_board_black_bg \
  --action_mode delta_eef_pose \
  --action_target absolute \
  --num_samples 6 \
  --max_steps 500 \
  --output_dir ./logs/action_viz_eef_abs