import traceback
import time
import os
import json
import math
import random
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import transformers

from data.filelock import FileLock
from data.hdf5_vla_dataset import HDF5VLADataset
from train.image_corrupt import image_corrupt


def get_clean_item(chunk_dir):
    """
    Get indexes of clean items in a chunk.
    """
    dirty_bit = read_dirty_bit(chunk_dir)
    return np.where(1 - dirty_bit)[0].tolist()


def save_dirty_bit(chunk_dir, dirty_bit):
    """
    Save the dirty bit to the chunk directory.
    """
    time_stmp = time.time()
    while time.time() - time_stmp < 10.0:
        try:
            file_path = os.path.join(chunk_dir, "dirty_bit")
            lock = FileLock(file_path)
            lock.acquire_write_lock()
            with open(file_path, 'wb') as file:
                file.write(dirty_bit.tobytes())
            lock.release_lock()
            return
        except KeyboardInterrupt:
            lock.release_lock()
            raise KeyboardInterrupt
        except BaseException:
            lock.release_lock()
            continue
    raise RuntimeError("Failed to save dirty bit.")


def read_dirty_bit(chunk_dir):
    """
    Read the dirty bit from the chunk directory.
    """
    # If error occurs, retry
    time_stmp = time.time()
    while time.time() - time_stmp < 10.0:
        try:
            file_path = os.path.join(chunk_dir, "dirty_bit")
            lock = FileLock(file_path)
            lock.acquire_read_lock()
            with open(file_path, 'rb') as file:
                dirty_bit = np.frombuffer(file.read(), dtype=np.uint8).copy()
            lock.release_lock()
            assert len(dirty_bit) > 0
            return dirty_bit
        except KeyboardInterrupt:
            lock.release_lock()
            raise KeyboardInterrupt
        except BaseException:
            lock.release_lock()
            continue
    raise RuntimeError("Failed to read dirty bit.")


class VLAConsumerDataset(Dataset):
    """A vision-languange-action Dataset for supervised training.
    This dataset will load data from the buffer directory.
    """
    
    def __init__(
        self, 
        config,
        tokenizer,
        image_processor,
        num_cameras,
        img_history_size,
        camera_views=None,
        image_size=None,
        auto_adjust_image_brightness=False,
        image_aug=False,
        dataset_type='pretrain',
        cond_mask_prob=0.1,
        cam_ext_mask_prob=-1.0,
        state_noise_snr=None,
        use_hdf5=False,
        hdf5_action_mode="delta_eef_pose",
        hdf5_action_target="delta",
        use_precomp_lang_embed=False
    ):
        super(VLAConsumerDataset, self).__init__()
        
        # Load the control frequency for each dataset
        with open("configs/dataset_control_freq.json", 'r') as fp:
            self.control_freq = json.load(fp)
        # Load the dataset names
        dataset_names_cfg = 'configs/pretrain_datasets.json' \
            if dataset_type == 'pretrain' else 'configs/finetune_datasets.json'
        with open(dataset_names_cfg, 'r') as file:
            DATASET_NAMES = json.load(file)
        # Create the mapping between dataset name and id
        self.dataset_name2id = {name: i for i, name in enumerate(DATASET_NAMES)}
        self.dataset_id2name = {i: name for i, name in enumerate(DATASET_NAMES)}
        
        self.image_processor = image_processor
        
        self.buffer_dir = config["buf_path"]
        self.num_chunks = config["buf_num_chunks"]
        self.chunk_size = config["buf_chunk_size"]
        self.tokenizer_max_length = config["tokenizer_max_length"]
        self.image_aspect_ratio = config["image_aspect_ratio"]
        self.state_noise_snr = state_noise_snr
        alias_to_view = {
            "head": "head",
            "cam_high": "head",
            "high": "head",
            "right": "right",
            "cam_right_wrist": "right",
            "right_wrist": "right",
            "left": "left",
            "cam_left_wrist": "left",
            "left_wrist": "left",
        }
        if camera_views is None:
            requested_views = ["head", "right", "left"]
        elif isinstance(camera_views, str):
            requested_views = [v.strip() for v in camera_views.split(",") if v.strip()]
        else:
            requested_views = [str(v).strip() for v in camera_views if str(v).strip()]
        if len(requested_views) == 0:
            raise ValueError("camera_views must contain at least one view")

        parsed_views = []
        for view in requested_views:
            key = view.lower()
            if key not in alias_to_view:
                raise ValueError(
                    f"Unsupported camera view '{view}'. Supported: head/right/left and cam_* aliases."
                )
            canonical = alias_to_view[key]
            if canonical not in parsed_views:
                parsed_views.append(canonical)

        self.num_cameras = int(num_cameras)
        if self.num_cameras < 1 or self.num_cameras > 3:
            raise ValueError("num_cameras must be in [1, 3]")
        self.camera_slot_order = ["head", "right", "left"][:self.num_cameras]

        for view in parsed_views:
            if view not in self.camera_slot_order:
                raise ValueError(
                    f"Camera view '{view}' is not available with num_cameras={self.num_cameras}"
                )
        self.enabled_camera_views = set(parsed_views)
        if len(self.enabled_camera_views) == 0:
            raise ValueError(
                "At least one selected camera view must be available under current num_cameras"
            )

        self._camera_slot_to_pair_idx = {"head": 0, "right": 1, "left": 2}
        self.img_history_size = img_history_size
        self.cond_mask_prob = cond_mask_prob
        self.cam_ext_mask_prob = cam_ext_mask_prob
        self.use_hdf5 = use_hdf5
        self.hdf5_action_mode = hdf5_action_mode
        self.hdf5_action_target = hdf5_action_target
        self.hdf5_dataset = None
        if use_hdf5:
            self.hdf5_dataset = HDF5VLADataset(
                action_mode=hdf5_action_mode,
                action_target=hdf5_action_target,
            )
        self.use_precomp_lang_embed = use_precomp_lang_embed
        if use_precomp_lang_embed:
            self.empty_lang_embed = torch.load("data/empty_lang_embed.pt")
        
        # Load dataset stat
        with open("configs/dataset_stat.json", 'r') as f:
            dataset_stat = json.load(f)
        self.dataset_stat = dataset_stat
        
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.auto_adjust_image_brightness = auto_adjust_image_brightness
        self.image_aug = image_aug
        
        self.last_content = None
        self.last_meta = None
    
    def get_dataset_name2id(self):
        return self.dataset_name2id
    
    def get_dataset_id2name(self):
        return self.dataset_id2name
        
    @staticmethod
    def pairwise(iterable):
        a = iter(iterable)
        return zip(a, a)

    @staticmethod
    def _load_data_from_chunk(chunk_dir, chunk_item_idx):
        # If error occurs, retry
        time_stmp = time.time()
        while time.time() - time_stmp < 10.0:
            try:
                locks = []
                file_path = os.path.join(chunk_dir, f"json_content_{chunk_item_idx}.json")
                lock = FileLock(file_path)
                locks.append(lock)
                lock.acquire_read_lock()
                with open(file_path, 'r') as file:
                    json_content = json.load(file)
                lock.release_lock()
                file_path = os.path.join(chunk_dir, f"sample_{chunk_item_idx}.npz")
                lock = FileLock(file_path)
                locks.append(lock)
                lock.acquire_read_lock()
                with open(file_path, 'rb') as file:
                    sample_dict = np.load(file)
                    meta = tuple(sample_dict.values())
                lock.release_lock()
                return json_content, meta
            except KeyboardInterrupt:
                for lock in locks:
                    lock.release_lock()
                raise KeyboardInterrupt
            except BaseException:
                for lock in locks:
                    lock.release_lock()
                continue
        raise RuntimeError("Failed to load sample.")

    def __len__(self) -> int:
        if self.use_hdf5:
            return len(self.hdf5_dataset)
        else:
            return self.num_chunks * self.chunk_size

    def _safe_load(self, index):
        read_chunk_item_indices = []
        # Start searching from a random chunk
        read_chunk_idx = index // self.chunk_size
        while len(read_chunk_item_indices) == 0:
            read_chunk_dir = os.path.join(self.buffer_dir, f"chunk_{read_chunk_idx}")
            try:
                read_chunk_item_indices = get_clean_item(read_chunk_dir)
            except BaseException as e:
                # Print the error info
                print("Error catched when searching a clean chunk:", e)
                traceback.print_exc()
                read_chunk_item_indices = []
            read_chunk_idx = (read_chunk_idx + 1) % self.num_chunks
        
        # read_chunk_item_index = random.choice(read_chunk_item_indices)
        # read_chunk_item_index = read_chunk_item_indices.pop()
        random_item_index = index % len(read_chunk_item_indices)
        read_chunk_item_index = read_chunk_item_indices[random_item_index]
        
        # Modify the dirty bit
        try:
            dirty_bit = read_dirty_bit(read_chunk_dir)
            dirty_bit[read_chunk_item_index] = 1
            save_dirty_bit(read_chunk_dir, dirty_bit)
        except BaseException as e:
            # Print the error info
            print("Error catched when modifying the dirty bit:", e)
            traceback.print_exc()
        
        # load the sample
        try:
            content, meta = self._load_data_from_chunk(read_chunk_dir, read_chunk_item_index)
            self.last_content, self.last_meta = content, meta
        except BaseException as e:
            # Print the error info
            print("Error catched when loading sample:", e)
            traceback.print_exc()
            
            # If failed to load the data, return the last loaded data for robustness
            content, meta = self.last_content, self.last_meta

        return (content, *meta)

    def _resolve_norm_stat(self, ds_stat, key_candidates, fallback, target_dim):
        for key in key_candidates:
            if key in ds_stat:
                arr = np.array(ds_stat[key], dtype=np.float32)
                if arr.shape[-1] == target_dim:
                    return arr
        if fallback is not None and fallback.shape[-1] == target_dim:
            return fallback.astype(np.float32)
        return None

    def _get_norm_stats(self, ds_name, state_dim, state_mean=None, state_std=None, action_mean=None, action_std=None):
        ds_stat = self.dataset_stat.get(ds_name, {})

        action_prefix = "absolute" if (self.use_hdf5 and self.hdf5_action_target == "absolute") else "delta"

        if self.use_hdf5 and self.hdf5_action_mode == "delta_joint":
            state_mean_keys = ["state_mean_delta_joint", "delta_joint_state_mean", "state_mean"]
            state_std_keys = ["state_std_delta_joint", "delta_joint_state_std", "state_std"]
            action_mean_keys = [
                f"action_mean_{action_prefix}_delta_joint",
                f"{action_prefix}_delta_joint_action_mean",
                f"action_mean_{action_prefix}",
                "action_mean_delta_joint",
                "delta_joint_action_mean",
                "action_mean",
            ]
            action_std_keys = [
                f"action_std_{action_prefix}_delta_joint",
                f"{action_prefix}_delta_joint_action_std",
                f"action_std_{action_prefix}",
                "action_std_delta_joint",
                "delta_joint_action_std",
                "action_std",
            ]
        elif self.use_hdf5 and self.hdf5_action_mode == "delta_eef_pose":
            state_mean_keys = ["state_mean_delta_eef_pose", "delta_eef_pose_state_mean", "state_mean"]
            state_std_keys = ["state_std_delta_eef_pose", "delta_eef_pose_state_std", "state_std"]
            action_mean_keys = [
                f"action_mean_{action_prefix}_delta_eef_pose",
                f"{action_prefix}_delta_eef_pose_action_mean",
                f"action_mean_{action_prefix}",
                "action_mean_delta_eef_pose",
                "delta_eef_pose_action_mean",
                "action_mean",
            ]
            action_std_keys = [
                f"action_std_{action_prefix}_delta_eef_pose",
                f"{action_prefix}_delta_eef_pose_action_std",
                f"action_std_{action_prefix}",
                "action_std_delta_eef_pose",
                "delta_eef_pose_action_std",
                "action_std",
            ]
        else:
            state_mean_keys = ["state_mean"]
            state_std_keys = ["state_std"]
            action_mean_keys = ["action_mean"]
            action_std_keys = ["action_std"]

        s_mean = self._resolve_norm_stat(ds_stat, state_mean_keys, state_mean, state_dim)
        s_std = self._resolve_norm_stat(ds_stat, state_std_keys, state_std, state_dim)
        a_mean = self._resolve_norm_stat(ds_stat, action_mean_keys, action_mean, state_dim)
        a_std = self._resolve_norm_stat(ds_stat, action_std_keys, action_std, state_dim)

        if s_mean is None:
            s_mean = np.zeros((state_dim,), dtype=np.float32)
        if s_std is None:
            s_std = np.ones((state_dim,), dtype=np.float32)
        if a_mean is None:
            a_mean = np.zeros((state_dim,), dtype=np.float32)
        if a_std is None:
            a_std = np.ones((state_dim,), dtype=np.float32)

        s_std = np.maximum(s_std, 1e-6)
        a_std = np.maximum(a_std, 1e-6)
        return s_mean, s_std, a_mean, a_std
    
    def __getitem__(self, index):
        # For robustness, we will try to load the data until we succeed
        while True:
            data_dict = None
            try:
                if self.use_hdf5:
                    res = self.hdf5_dataset.get_item()
                    content = res['meta']
                    states = res['state']
                    actions = res['actions']
                    state_elem_mask = res['state_indicator']
                    image_metas = [
                        res['cam_high'], res['cam_high_mask'],
                        res['cam_right_wrist'], res['cam_right_wrist_mask'],
                        res['cam_left_wrist'], res['cam_left_wrist_mask'],
                    ]
                    state_std = res['state_std']
                    state_mean = res['state_mean']
                    action_std = res.get('action_std')
                    action_mean = res.get('action_mean')
                    state_norm = res['state_norm']
                else:
                    (content, _, states, _, actions, _, 
                    state_elem_mask, *image_metas, 
                    state_std, state_mean, state_norm) = self._safe_load(index)
                    action_std = None
                    action_mean = None
                
                data_dict = {}
                data_dict['dataset_name'] = content['dataset_name']
                data_dict['data_idx'] = self.dataset_name2id[data_dict['dataset_name']]
                data_dict['ctrl_freq'] = self.control_freq[data_dict['dataset_name']] \
                    if random.random() > self.cond_mask_prob else 0
                
                s_mean, s_std, a_mean, a_std = self._get_norm_stats(
                    ds_name=data_dict['dataset_name'],
                    state_dim=states.shape[-1],
                    state_mean=state_mean,
                    state_std=state_std,
                    action_mean=action_mean,
                    action_std=action_std,
                )

                states = (states - s_mean[None]) / s_std[None]
                actions = (actions - a_mean[None]) / a_std[None]
                states = states * state_elem_mask[None]
                actions = actions * state_elem_mask[None]

                if self.state_noise_snr is not None:
                    # Add noise after normalization so state scales are comparable across dimensions.
                    states += np.random.normal(
                        0.0,
                        1.0 / np.sqrt(10 ** (self.state_noise_snr / 10)),
                        states.shape,
                    )

                # ds_state_mean was used for masking. In normalized space, the mean is 0.
                ds_state_mean = np.zeros_like(states)
                
                # Randomly mask the states by the mean state (0)
                data_dict["states"] = states \
                    if random.random() > self.cond_mask_prob else ds_state_mean
                data_dict["actions"] = actions
                data_dict["state_elem_mask"] = state_elem_mask \
                    if random.random() > self.cond_mask_prob else np.zeros_like(state_elem_mask)
                
                # Stat for the episode that the step belongs to 
                data_dict["state_norm"] = state_norm
                
                # We replace the invalid images with the background image
                # and also randomly mask images by the background image
                background_color = np.array([
                    int(x*255) for x in self.image_processor.image_mean
                ], dtype=np.uint8).reshape(1, 1, 3)
                background_image = np.ones((
                    self.image_processor.size["height"], 
                    self.image_processor.size["width"], 3), dtype=np.uint8
                ) * background_color
                
                image_metas = list(self.pairwise(image_metas))
                mask_probs = [self.cond_mask_prob] * self.num_cameras
                if self.cam_ext_mask_prob >= 0.0 and "head" in self.camera_slot_order:
                    head_idx = self.camera_slot_order.index("head")
                    mask_probs[head_idx] = self.cam_ext_mask_prob
                rearranged_images = []
                for i in range(self.img_history_size):
                    for j in range(self.num_cameras):
                        slot_view = self.camera_slot_order[j]
                        if slot_view not in self.enabled_camera_views:
                            rearranged_images.append((background_image.copy(), False))
                            continue
                        images, image_mask = image_metas[j]
                        image, valid = images[i], image_mask[i]
                        if valid and (math.prod(image.shape) > 0) and \
                            (random.random() > mask_probs[j]):
                            rearranged_images.append((image, True))
                        else:
                            rearranged_images.append((background_image.copy(), False))
                
                preprocessed_images = []
                processor = self.image_processor
                for image, valid in rearranged_images:
                    image = Image.fromarray(image)
                    if self.image_size is not None:
                        image = transforms.Resize(self.image_size)(image) # (1008, 336)
                    # assert image.height == 336, "We haven't prepare for training with images of different resolutions."
                    
                    if valid and self.auto_adjust_image_brightness:
                        pixel_values = list(image.getdata())
                        average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                        if average_brightness <= 0.15:
                            image = transforms.ColorJitter(brightness=(1.75,1.75))(image)
                    
                    # Only apply image augmentation to 50% of the images
                    if valid and self.image_aug and (random.random() > 0.5):
                        aug_type = random.choice([
                            "corrput_only", "color_only", "both"])
                        if aug_type != "corrput_only":
                            image = transforms.ColorJitter(
                                brightness=0.3, contrast=0.4, saturation=0.5, hue=0.03)(image)
                        if aug_type != "color_only":
                            image = image_corrupt(image)
                    
                    if self.image_aspect_ratio == 'pad':
                        def expand2square(pil_img, background_color):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result
                        image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    preprocessed_images.append(image)
                data_dict["images"] = preprocessed_images

                if self.use_precomp_lang_embed:
                    if content["instruction"][-1] == ".":
                        content["instruction"] = content["instruction"][:-1]
                    data_dict["lang_embed"] = torch.load(content["instruction"]) \
                        if random.random() > self.cond_mask_prob else self.empty_lang_embed
                else:
                    instruction = content["instruction"] \
                        if random.random() > self.cond_mask_prob else ""
                    data_dict["input_ids"] = self.tokenizer(
                        instruction,
                        return_tensors="pt",
                        padding="longest",
                        truncation=False,
                    ).input_ids[0]
                
                    assert len(data_dict["input_ids"]) <= self.tokenizer_max_length, \
                        f"Instruction length {len(data_dict['input_ids'])} exceeds the maximum length {self.tokenizer_max_length}."
                
                for k, v in data_dict.items():
                    if isinstance(v, np.ndarray):
                        data_dict[k] = torch.from_numpy(v)

                for k, v in data_dict.items():
                    assert not isinstance(v, np.ndarray), f"key: {k}, value: {v}"
                        # data_dict[k] = torch.from_numpy(v)
        
                return data_dict
            except BaseException as e:
                # Print the error info
                if data_dict is not None:
                    print(f"Error catched when processing sample from {data_dict.get('dataset_name')}:", e)
                else:
                    print(f"Error catched when processing sample:", e)
                traceback.print_exc()
                # Try incresing the index
                index = (index + 1) % len(self)


class DataCollatorForVLAConsumerDataset(object):
    """Collate examples for supervised training."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = {
            "states": [],
            "actions": [],
            "state_elem_mask": [],
            "state_norm": [],
            "images": [],
            "data_indices": [],
            "ctrl_freqs": []
        }
        input_ids = []
        lang_embeds = []
        lang_embed_lens = []
        
        for instance in instances:
            # Convert all the numpy arrays to tensor
            keys_to_check = [
                'states', 'actions',
                'state_elem_mask', 'state_norm',
            ]
            for key in keys_to_check:
                if isinstance(instance[key], torch.Tensor):
                    item = instance[key]
                else:
                    item = torch.from_numpy(instance[key])
                batch[key].append(item)

            if "input_ids" in instance:
                input_ids.append(instance["input_ids"])
            else:
                lang_embeds.append(instance["lang_embed"])
                lang_embed_lens.append(instance["lang_embed"].shape[0])
            
            batch["images"].append(torch.stack(instance["images"], dim=0))
            batch["data_indices"].append(instance["data_idx"])
            batch["ctrl_freqs"].append(instance["ctrl_freq"])
        
        keys_to_stack = [
            'states', 'actions',
            'state_elem_mask', 'state_norm',
            "images"
        ]
        for key in keys_to_stack:
            batch[key] = torch.stack(batch[key], dim=0)
        
        batch["ctrl_freqs"] = torch.tensor(batch["ctrl_freqs"])
    
        if len(input_ids) > 0:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            batch["input_ids"] = input_ids
            batch["lang_attn_mask"] = input_ids.ne(self.tokenizer.pad_token_id)
        else:
            lang_embeds = torch.nn.utils.rnn.pad_sequence(
                lang_embeds,
                batch_first=True,
                padding_value=0)
            input_lang_attn_mask = torch.zeros(
                lang_embeds.shape[0], lang_embeds.shape[1], dtype=torch.bool)
            for i, l in enumerate(lang_embed_lens):
                input_lang_attn_mask[i, :l] = True
            batch["lang_embeds"] = lang_embeds
            batch["lang_attn_mask"] = input_lang_attn_mask
            
            
        return batch
