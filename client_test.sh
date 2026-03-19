python scripts/rdt_mock_client.py \
    --server_url http://127.0.0.1:18081 \
    --dataset_path /media/damoxing/datasets/myendless/pick_up_board_black_bg \
    --action_mode eef_pose \
    --action_target delta \
    --output_video logs/rdt_eval_full_dataset.mp4 \
    --video_fps 20 --max_episodes 1