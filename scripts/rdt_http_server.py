#!/usr/bin/env python
# coding=utf-8

import argparse
import base64
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from models.rdt_runner import RDTRunner
from scripts.action_mode_utils import build_state_mask, get_active_indices


class RDTService:
    def __init__(
        self,
        config_path,
        pretrained_model_name_or_path,
        pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path,
        dataset_stat_path,
        dataset_name,
        action_mode,
        action_target,
        control_frequency,
        device,
        dtype,
    ):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        with open(dataset_stat_path, "r") as f:
            self.dataset_stat = json.load(f)

        self.dataset_name = dataset_name
        self.action_mode = action_mode
        self.action_target = action_target
        self.state_dim = self.cfg["common"]["state_dim"]
        self.control_frequency = int(control_frequency)
        self.device = torch.device(device)
        self.dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[dtype]

        self.active_indices = get_active_indices(self.action_mode)
        self.state_mask_np = build_state_mask(self.state_dim, self.active_indices)

        text_embedder = T5Embedder(
            from_pretrained=pretrained_text_encoder_name_or_path,
            model_max_length=self.cfg["dataset"]["tokenizer_max_length"],
            device=self.device,
        )
        self.text_tokenizer = text_embedder.tokenizer
        self.text_encoder = text_embedder.model.to(self.device, dtype=self.dtype).eval()

        self.vision_encoder = SiglipVisionTower(
            vision_tower=pretrained_vision_encoder_name_or_path,
            args=None,
        )
        self.image_processor = self.vision_encoder.image_processor
        self.vision_encoder = self.vision_encoder.to(self.device, dtype=self.dtype).eval()

        self.policy = self._build_policy(pretrained_model_name_or_path)
        self.policy = self.policy.to(self.device, dtype=self.dtype).eval()

        self._lang_cache = {}

    def _build_policy(self, pretrained):
        if pretrained is not None and os.path.isdir(pretrained):
            return RDTRunner.from_pretrained(pretrained)

        img_cond_len = (
            self.cfg["common"]["img_history_size"]
            * self.cfg["common"]["num_cameras"]
            * self.vision_encoder.num_patches
        )
        model = RDTRunner(
            action_dim=self.cfg["common"]["state_dim"],
            pred_horizon=self.cfg["common"]["action_chunk_size"],
            config=self.cfg["model"],
            lang_token_dim=self.cfg["model"]["lang_token_dim"],
            img_token_dim=self.cfg["model"]["img_token_dim"],
            state_token_dim=self.cfg["model"]["state_token_dim"],
            max_lang_cond_len=self.cfg["dataset"]["tokenizer_max_length"],
            img_cond_len=img_cond_len,
            img_pos_embed_config=[
                (
                    "image",
                    (
                        self.cfg["common"]["img_history_size"],
                        self.cfg["common"]["num_cameras"],
                        -self.vision_encoder.num_patches,
                    ),
                )
            ],
            lang_pos_embed_config=[("lang", -self.cfg["dataset"]["tokenizer_max_length"])],
            dtype=self.dtype,
        )

        if pretrained is not None and os.path.isfile(pretrained):
            filename = os.path.basename(pretrained)
            if filename.endswith(".pt"):
                checkpoint = torch.load(pretrained, map_location="cpu")
                model.load_state_dict(checkpoint["module"])
            elif filename.endswith(".safetensors"):
                from safetensors.torch import load_model

                load_model(model, pretrained)
            else:
                raise NotImplementedError(f"Unknown checkpoint format: {pretrained}")

        return model

    def _pick_stat(self, ds_stat, keys, fallback):
        for key in keys:
            if key in ds_stat:
                arr = np.array(ds_stat[key], dtype=np.float32)
                if arr.shape[-1] == self.state_dim:
                    return arr
        return fallback

    def _get_norm_stats(self):
        ds_stat = self.dataset_stat.get(self.dataset_name, {})
        action_prefix = "absolute" if self.action_target == "absolute" else "delta"

        if self.action_mode == "joint":
            state_mean_keys = ["state_mean_joint", "joint_state_mean", "state_mean_delta_joint", "delta_joint_state_mean", "state_mean"]
            state_std_keys = ["state_std_joint", "joint_state_std", "state_std_delta_joint", "delta_joint_state_std", "state_std"]
            action_mean_keys = [
                f"action_mean_{action_prefix}_joint",
                f"{action_prefix}_joint_action_mean",
                f"action_mean_{action_prefix}_delta_joint",
                f"{action_prefix}_delta_joint_action_mean",
                f"action_mean_{action_prefix}",
                "action_mean_joint",
                "joint_action_mean",
                "action_mean_delta_joint",
                "delta_joint_action_mean",
                "action_mean",
            ]
            action_std_keys = [
                f"action_std_{action_prefix}_joint",
                f"{action_prefix}_joint_action_std",
                f"action_std_{action_prefix}_delta_joint",
                f"{action_prefix}_delta_joint_action_std",
                f"action_std_{action_prefix}",
                "action_std_joint",
                "joint_action_std",
                "action_std_delta_joint",
                "delta_joint_action_std",
                "action_std",
            ]
        else:
            state_mean_keys = ["state_mean_eef_pose", "eef_pose_state_mean", "state_mean_delta_eef_pose", "delta_eef_pose_state_mean", "state_mean"]
            state_std_keys = ["state_std_eef_pose", "eef_pose_state_std", "state_std_delta_eef_pose", "delta_eef_pose_state_std", "state_std"]
            action_mean_keys = [
                f"action_mean_{action_prefix}_eef_pose",
                f"{action_prefix}_eef_pose_action_mean",
                f"action_mean_{action_prefix}_delta_eef_pose",
                f"{action_prefix}_delta_eef_pose_action_mean",
                f"action_mean_{action_prefix}",
                "action_mean_eef_pose",
                "eef_pose_action_mean",
                "action_mean_delta_eef_pose",
                "delta_eef_pose_action_mean",
                "action_mean",
            ]
            action_std_keys = [
                f"action_std_{action_prefix}_eef_pose",
                f"{action_prefix}_eef_pose_action_std",
                f"action_std_{action_prefix}_delta_eef_pose",
                f"{action_prefix}_delta_eef_pose_action_std",
                f"action_std_{action_prefix}",
                "action_std_eef_pose",
                "eef_pose_action_std",
                "action_std_delta_eef_pose",
                "delta_eef_pose_action_std",
                "action_std",
            ]

        s_mean = self._pick_stat(
            ds_stat,
            state_mean_keys,
            np.zeros(self.state_dim, dtype=np.float32),
        )
        s_std = self._pick_stat(
            ds_stat,
            state_std_keys,
            np.ones(self.state_dim, dtype=np.float32),
        )
        a_mean = self._pick_stat(
            ds_stat,
            action_mean_keys,
            np.zeros(self.state_dim, dtype=np.float32),
        )
        a_std = self._pick_stat(
            ds_stat,
            action_std_keys,
            np.ones(self.state_dim, dtype=np.float32),
        )

        s_std = np.maximum(s_std, 1e-6)
        a_std = np.maximum(a_std, 1e-6)
        return s_mean, s_std, a_mean, a_std

    def _decode_image(self, b64_str):
        import cv2

        if b64_str is None:
            return None
        raw = base64.b64decode(b64_str.encode("utf-8"))
        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img

    def _prepare_images(self, images_payload):
        bg_color = np.array(
            [int(x * 255) for x in self.image_processor.image_mean],
            dtype=np.uint8,
        ).reshape(1, 1, 3)
        bg = np.ones(
            (
                self.image_processor.size["height"],
                self.image_processor.size["width"],
                3,
            ),
            dtype=np.uint8,
        ) * bg_color

        cam_high = images_payload.get("cam_high", [])
        cam_right = images_payload.get("cam_right_wrist", [])
        cam_left = images_payload.get("cam_left_wrist", [])
        hist = self.cfg["common"]["img_history_size"]

        if not (len(cam_high) == len(cam_right) == len(cam_left) == hist):
            raise ValueError(f"Each camera must provide exactly {hist} frames")

        ordered = []
        for i in range(hist):
            ordered.append(self._decode_image(cam_high[i]))
            ordered.append(self._decode_image(cam_right[i]))
            ordered.append(self._decode_image(cam_left[i]))

        image_tensors = []
        for arr in ordered:
            if arr is None:
                pil_img = Image.fromarray(bg)
            else:
                pil_img = Image.fromarray(arr)

            if self.cfg["dataset"].get("image_aspect_ratio", "pad") == "pad":
                w, h = pil_img.size
                if w != h:
                    side = max(w, h)
                    result = Image.new(pil_img.mode, (side, side), tuple(int(x * 255) for x in self.image_processor.image_mean))
                    result.paste(pil_img, ((side - w) // 2, (side - h) // 2))
                    pil_img = result

            t = self.image_processor.preprocess(pil_img, return_tensors="pt")["pixel_values"][0]
            image_tensors.append(t)

        image_tensor = torch.stack(image_tensors, dim=0).to(self.device, dtype=self.dtype)
        img_embeds = self.vision_encoder(image_tensor).detach()
        img_embeds = img_embeds.reshape(1, -1, self.vision_encoder.hidden_size)
        return img_embeds

    @torch.no_grad()
    def infer(self, request_data):
        instruction = request_data["instruction"]
        state = np.array(request_data["state"], dtype=np.float32)
        images = request_data["images"]
        ctrl_freq = int(request_data.get("ctrl_freq", self.control_frequency))

        if state.shape[-1] != self.state_dim:
            raise ValueError(f"state length must be {self.state_dim}, got {state.shape[-1]}")

        s_mean, s_std, a_mean, a_std = self._get_norm_stats()
        state_norm = (state - s_mean) / s_std
        state_norm = state_norm * self.state_mask_np

        state_tensor = torch.from_numpy(state_norm).view(1, 1, -1).to(self.device, dtype=self.dtype)
        state_mask = torch.from_numpy(self.state_mask_np).view(1, -1).to(self.device, dtype=self.dtype)

        img_embeds = self._prepare_images(images)

        if instruction in self._lang_cache:
            text_embeds = self._lang_cache[instruction]
        else:
            tokens = self.text_tokenizer(
                instruction,
                return_tensors="pt",
                padding="longest",
                truncation=False,
            )["input_ids"].to(self.device)
            text_embeds = self.text_encoder(tokens).last_hidden_state.detach().to(self.dtype)
            self._lang_cache[instruction] = text_embeds

        lang_mask = torch.ones(text_embeds.shape[:2], dtype=torch.bool, device=self.device)
        ctrl_freqs = torch.tensor([ctrl_freq], dtype=torch.int64, device=self.device)

        pred_norm = self.policy.predict_action(
            lang_tokens=text_embeds,
            lang_attn_mask=lang_mask,
            img_tokens=img_embeds,
            state_tokens=state_tensor,
            action_mask=state_mask.unsqueeze(1),
            ctrl_freqs=ctrl_freqs,
        )
        pred_norm = pred_norm.to(torch.float32).squeeze(0).cpu().numpy()

        pred = pred_norm * a_std[None, :] + a_mean[None, :]
        pred = pred * self.state_mask_np[None, :]
        return pred, pred_norm


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, code, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._send_json(
                    200,
                    {
                        "ok": True,
                        "action_mode": service.action_mode,
                        "action_target": service.action_target,
                        "state_dim": service.state_dim,
                        "active_indices": service.active_indices,
                    },
                )
            return self._send_json(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            if self.path != "/infer":
                return self._send_json(404, {"ok": False, "error": "not found"})

            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                req = json.loads(raw.decode("utf-8"))
                t0 = time.time()
                pred, pred_norm = service.infer(req)
                dt = (time.time() - t0) * 1000.0
                return self._send_json(
                    200,
                    {
                        "ok": True,
                        "action": pred.tolist(),
                        "action_normalized": pred_norm.tolist(),
                        "action_mode": service.action_mode,
                        "action_target": service.action_target,
                        "active_indices": service.active_indices,
                        "latency_ms": dt,
                    },
                )
            except Exception as e:
                return self._send_json(400, {"ok": False, "error": str(e)})

    return Handler


def parse_args():
    parser = argparse.ArgumentParser("RDT REST inference server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config_path", type=str, default="configs/base.yaml")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--pretrained_text_encoder_name_or_path", type=str, default="google/t5-v1_1-xxl")
    parser.add_argument("--pretrained_vision_encoder_name_or_path", type=str, default="google/siglip-so400m-patch14-384")
    parser.add_argument("--dataset_stat_path", type=str, default="configs/dataset_stat.json")
    parser.add_argument("--dataset_name", type=str, default="astribot")
    parser.add_argument("--action_mode", type=str, default="eef_pose", choices=["joint", "eef_pose"])
    parser.add_argument("--action_target", type=str, default="delta", choices=["delta", "absolute"])
    parser.add_argument("--control_frequency", type=int, default=25)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    return parser.parse_args()


def main():
    args = parse_args()
    import debugpy; debugpy.listen(("localhost", 5678)); print("Waiting for debugger attach..."); debugpy.wait_for_client()
    service = RDTService(
        config_path=args.config_path,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        pretrained_text_encoder_name_or_path=args.pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path=args.pretrained_vision_encoder_name_or_path,
        dataset_stat_path=args.dataset_stat_path,
        dataset_name=args.dataset_name,
        action_mode=args.action_mode,
        action_target=args.action_target,
        control_frequency=args.control_frequency,
        device=args.device,
        dtype=args.dtype,
    )

    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(
        f"RDT server listening at http://{args.host}:{args.port} "
        f"action_mode={args.action_mode} action_target={args.action_target}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
