import os, sys

now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)

from huggingface_hub import snapshot_download

ckpt_dir = os.path.join(now_dir, "checkpoints", "fish-speech-1.5")
from pathlib import Path
import torch
from tools.vqgan.inference import load_model as load_decoder_model
from tools.llama.generate import encode_tokens, load_model, generate_long
import folder_paths
import json
import io
import torchaudio
from comfy.cli_args import args
from vqgan_utils import load_model as load_vqgan_model
from vqgan_utils import audio2prompt, semantic2audio
from llama_utils import prompt2semantic

CKPTS_FOLDER = os.path.join(now_dir, "checkpoints")
CONFIGS_FOLDER = os.path.join(now_dir, "fish_speech", "configs")


class LoadVQGAN:
    def __init__(self):
        self.vqgan = None
        pass

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "config": (
                    [
                        str(c.relative_to(CONFIGS_FOLDER))
                        for c in Path(CONFIGS_FOLDER).glob("*vq*.yaml")
                    ],
                    {"default": "firefly_gan_vq.yaml"},
                ),
                "model": (
                    [
                        str(p.relative_to(CKPTS_FOLDER))
                        for p in Path(CKPTS_FOLDER).glob("**/*vq*.pth")
                    ],
                ),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
        }

    @classmethod
    def IS_CHANGED(s, model):
        return ""

    @classmethod
    def VALIDATE_INPUTS(s, model):
        return True

    RETURN_TYPES = ("VQGAN",)
    RETURN_NAMES = ("vqgan",)

    FUNCTION = "load_vqgan"

    # OUTPUT_NODE = False

    CATEGORY = "FishSpeech_EX"

    def load_vqgan(self, config, model, device):
        config = config.rsplit(".", 1)[0]
        model = str(CKPTS_FOLDER + "\\" + model)
        if self.vqgan is None:
            self.vqgan = load_vqgan_model(config, model, device=device)
        return (self.vqgan,)


class AudioToPrompt:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "vqgan": ("VQGAN",),
                "audio": ("AUDIO",),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
        }

    RETURN_TYPES = ("AUDIO", "NUMPY")
    RETURN_NAMES = ("restored_audio", "prompt_tokens")

    FUNCTION = "encode"

    # OUTPUT_NODE = False

    CATEGORY = "FishSpeech_EX"

    def encode(self, vqgan, audio, device):
        return audio2prompt(vqgan, audio, device)


class Prompt2Semantic:
    def __init__(self):
        if not os.path.exists(os.path.join(ckpt_dir, "model.pth")):
            snapshot_download(repo_id="fishaudio/fish-speech-1.5", local_dir=ckpt_dir)
        else:
            print("use cached model weights!")
        self.vqgan_model = None
        self.llama_model = None

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "precision": (["bf16", "half"], {"default": "bf16"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "text": ("STRING", {"multiline": True}),
                "prompt_text": ("STRING", {"multiline": True}),
                "prompt_tokens": ("NUMPY",),
                "num_samples": (
                    "INT",
                    {"default": 1, "min": 1, "max": 10, "step": 1, "display": "number"},
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2048,
                        "step": 8,
                        "display": "number",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.1,
                        "display": "number",
                    },
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 1.0,
                        "max": 2.0,
                        "step": 0.1,
                        "display": "number",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.1,
                        "display": "number",
                    },
                ),
                "compile": (["yes", "no"], {"default": "no"}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4294967295,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "iterative_prompt": (["yes", "no"], {"default": "yes"}),
                "chunk_length": (
                    "INT",
                    {
                        "default": 100,
                        "min": 0,
                        "max": 500,
                        "step": 8,
                        "display": "number",
                    },
                ),
            },
        }

    RETURN_TYPES = ("NUMPY",)
    RETURN_NAMES = ("codes",)

    FUNCTION = "decode"

    # OUTPUT_NODE = False

    CATEGORY = "FishSpeech_EX"

    def decode(
        self,
        precision: str,
        device: str,
        text: str,
        prompt_text: str,
        prompt_tokens,
        num_samples: int,
        max_new_tokens: int,
        top_p: float,
        repetition_penalty: float,
        temperature: float,
        compile: str,
        seed: int,
        iterative_prompt: str,
        chunk_length: int,
    ):
        precision = torch.bfloat16 if precision == "bf16" else torch.half

        if self.llama_model is None:
            print("Loading lla model ...")
            self.llama_model, self.decode_one_token = load_model(
                ckpt_dir, device, precision, compile=True if compile == "yes" else False
            )
            with torch.device(device):
                self.llama_model.setup_caches(
                    max_batch_size=1,
                    max_seq_len=self.llama_model.config.max_seq_len,
                    dtype=next(self.llama_model.parameters()).dtype,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        return prompt2semantic(
            self.llama_model,
            self.decode_one_token,
            text,
            num_samples,
            [
                prompt_text,
            ],
            [
                prompt_tokens,
            ],
            max_new_tokens,
            top_p,
            repetition_penalty,
            temperature,
            device,
            compile=True if compile == "yes" else False,
            seed=seed,
            iterative_prompt=True if iterative_prompt == "yes" else False,
            chunk_length=chunk_length,
        )


class Semantic2Audio:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vqgan": ("VQGAN",),
                "codes": ("NUMPY",),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("generated_audio",)

    FUNCTION = "generate"

    # OUTPUT_NODE = False

    CATEGORY = "FishSpeech_EX"

    def generate(self, vqgan, codes, device):
        return semantic2audio(vqgan, codes, device)


class SaveAudioToMp3:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_audio"

    OUTPUT_NODE = True

    CATEGORY = "FishSpeech_EX"

    def save_audio(
        self, audio, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None
    ):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        )
        results = list()

        metadata = {}
        if not args.disable_metadata:
            if prompt is not None:
                metadata["prompt"] = json.dumps(prompt)
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata[x] = json.dumps(extra_pnginfo[x])

        for batch_number, waveform in enumerate(audio["waveform"].cpu()):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.mp3"

            buff = io.BytesIO()
            torchaudio.save(buff, waveform, audio["sample_rate"], format="MP3")

            # buff = insert_or_replace_vorbis_comment(buff, metadata) 本质上是写入工作流，Mp3不适合

            with open(os.path.join(full_output_folder, file), "wb") as f:
                f.write(buff.getbuffer())

            results.append(
                {"filename": file, "subfolder": subfolder, "type": self.type}
            )
            counter += 1

        return {"ui": {"audio": results}}


NODE_CLASS_MAPPINGS = {
    "LoadVQGAN": LoadVQGAN,
    "AudioToPrompt": AudioToPrompt,
    "Prompt2Semantic": Prompt2Semantic,
    "Semantic2Audio": Semantic2Audio,
    "SaveAudioToMp3": SaveAudioToMp3,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadVQGAN": "EX_LoadVQGAN",
    "AudioToPrompt": "EX_AudioToPrompt",
    "Prompt2Semantic": "EX_Prompt2Semantic",
    "Semantic2Audio": "EX_Semantic2Audio",
    "SaveAudioToMp3": "EX_SaveAudioToMp3",
}
