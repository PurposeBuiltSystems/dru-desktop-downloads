#!/usr/bin/env python3
"""Pack local reconstruction weights into grouped four-bit storage and load them safely."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np

FORMAT = "groupwise-affine-int4"
VERSION = 1


def pack_groups(values, group_size):
    if group_size not in [16, 32, 64, 128]:
        raise ValueError("Unsupported four-bit group size")
    flat = np.asarray(values, dtype=np.float32).ravel()
    if not len(flat) or not np.isfinite(flat).all():
        raise ValueError("Weights must contain finite values")
    padding = (-len(flat)) % group_size
    groups = np.pad(flat, (0, padding)).reshape(-1, group_size)
    lower = np.minimum(groups.min(axis=1), 0)
    upper = np.maximum(groups.max(axis=1), 0)
    best_error = np.full(len(groups), np.inf)
    best_quantized = np.zeros(groups.shape, np.uint8)
    best_scale = np.zeros(len(groups), np.float16)
    best_offset = np.zeros(len(groups), np.float16)
    for clipping in [1., .975, .95, .9]:
        offset = (lower * clipping).astype(np.float16)
        scale = np.maximum((upper - lower) * clipping / 15, 1e-7).astype(np.float16)
        if not np.isfinite(offset).all() or not np.isfinite(scale).all():
            raise ValueError("Weight range exceeds the compact scale representation")
        quantized = np.rint((groups - offset.astype(np.float32)[:, None]) / scale.astype(np.float32)[:, None]).clip(0, 15).astype(np.uint8)
        restored = quantized.astype(np.float32) * scale.astype(np.float32)[:, None] + offset.astype(np.float32)[:, None]
        error = np.mean((restored - groups) ** 2, axis=1)
        selected = error < best_error
        best_error[selected] = error[selected]
        best_quantized[selected] = quantized[selected]
        best_scale[selected] = scale[selected]
        best_offset[selected] = offset[selected]
    quantized = best_quantized.ravel()
    packed = quantized[::2] | (quantized[1::2] << 4)
    return packed, best_scale, best_offset


def unpack_groups(packed, scales, offsets, shape, group_size):
    if group_size not in [16, 32, 64, 128]:
        raise ValueError("Unsupported four-bit group size")
    count = math.prod(shape)
    groups = math.ceil(count / group_size)
    if packed.dtype != np.uint8 or packed.size != groups * group_size // 2 or scales.size != groups or offsets.size != groups:
        raise ValueError("Invalid packed weight dimensions")
    if not np.isfinite(scales).all() or not np.isfinite(offsets).all() or (scales <= 0).any():
        raise ValueError("Invalid packed weight scales")
    values = np.empty(packed.size * 2, dtype=np.float32)
    values[::2] = packed & 15
    values[1::2] = packed >> 4
    values = values.reshape(groups, group_size)
    values *= scales.astype(np.float32)[:, None]
    values += offsets.astype(np.float32)[:, None]
    return values.ravel()[:count].reshape(shape)


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(source, output, group_size, license_path):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    torch.set_num_threads(4)
    if output.exists():
        raise FileExistsError("Choose a new package folder")
    license_text = license_path.read_text()
    if "Apache License" not in license_text:
        raise ValueError("Supply the Apache-2.0 model license")
    checkpoint = source / "model.safetensors"
    tensors = {}
    weights = {}
    quantized_parameters = 0
    with safe_open(checkpoint, framework="pt", device="cpu") as original:
        stored_names = set(original.keys())
        aliases = {alias: target for alias, target in (original.metadata() or {}).items() if target in stored_names}
        for name in original.keys():
            tensor = original.get_tensor(name)
            shape = list(tensor.shape)
            if name.endswith(".weight") and len(shape) >= 2 and tensor.numel() >= 4096 and tensor.dtype == torch.float32:
                packed, scales, offsets = pack_groups(tensor.numpy(), group_size)
                tensors[name + ".packed"] = torch.from_numpy(packed)
                tensors[name + ".scales"] = torch.from_numpy(scales)
                tensors[name + ".offsets"] = torch.from_numpy(offsets)
                weights[name] = {"shape": shape, "storage": "int4"}
                quantized_parameters += tensor.numel()
            else:
                tensors[name] = tensor.contiguous()
                weights[name] = {"shape": shape, "storage": "original"}
    output.mkdir(parents=True)
    temporary = output / "weights.safetensors.partial"
    save_file(tensors, str(temporary), metadata={"format": FORMAT, "version": str(VERSION)})
    temporary.replace(output / "weights.safetensors")
    shutil.copy2(source / "config.json", output / "config.json")
    (output / "LICENSE.txt").write_text(license_text)
    metadata = {"format": FORMAT, "version": VERSION, "group_size": group_size, "source": "facebook/map-anything-apache", "source_checkpoint_sha256": file_digest(checkpoint), "weights_sha256": file_digest(output / "weights.safetensors"), "quantized_parameters": quantized_parameters, "aliases": aliases, "weights": weights, "runtime": "Packed storage. Weights expand into float32 memory for the existing PyTorch CPU/MPS/CUDA pipeline."}
    (output / "quantization.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "NOTICE.txt").write_text("Derived from facebook/map-anything-apache, licensed Apache-2.0. Modified through grouped four-bit weight quantization. Model output accuracy and hardware compatibility require validation.\n" + metadata["runtime"] + "\n")
    print(json.dumps({"package": str(output), "bytes": sum(p.stat().st_size for p in output.iterdir() if p.is_file()), "quantized_parameters": quantized_parameters}, indent=2))


def load_model(folder, model_class, native_int4=False):
    import torch
    from safetensors import safe_open
    folder = Path(folder)
    metadata = json.loads((folder / "quantization.json").read_text())
    if metadata["format"] != FORMAT or metadata["version"] != VERSION:
        raise ValueError("Unsupported quantized model format")
    checkpoint = folder / "weights.safetensors"
    if file_digest(checkpoint) != metadata["weights_sha256"]:
        raise ValueError("Quantized model failed its integrity check")
    config = json.loads((folder / "config.json").read_text())
    config["encoder_config"]["torch_hub_pretrained"] = False
    config["pretrained_checkpoint_path"] = None
    config["load_specific_pretrained_submodules"] = False
    with torch.device("meta"):
        model = model_class(**config)
    expected = model.state_dict()
    references = dict(model.named_parameters(remove_duplicate=False))
    references.update(model.named_buffers(remove_duplicate=False))
    aliases = metadata.get("aliases", {})
    stored_names = metadata["weights"].keys()
    available = set(stored_names) | set(aliases)
    if expected.keys() != available:
        raise ValueError("Model configuration differs from packed weights")
    for alias, original in aliases.items():
        if original not in stored_names or references[alias] is not references[original] or expected[alias].shape != expected[original].shape:
            raise ValueError("Packed aliases must match shared model parameters")
    restored = {}
    replacements = {}
    if native_int4:
        from metal_weights import MetalLinear
    with safe_open(checkpoint, framework="pt", device="cpu") as weights, torch.no_grad():
        for name, specification in metadata["weights"].items():
            if list(expected[name].shape) != specification["shape"]:
                raise ValueError(f"Unexpected tensor shape for {name}")
            module_path = name.removesuffix(".weight")
            linear = model.get_submodule(module_path) if name.endswith(".weight") else None
            if native_int4 and specification["storage"] == "int4" and isinstance(linear, torch.nn.Linear) and specification["shape"][1] % metadata["group_size"] == 0:
                replacement = MetalLinear(weights.get_tensor(name + ".packed").numpy(), weights.get_tensor(name + ".scales").numpy(), weights.get_tensor(name + ".offsets").numpy(), specification["shape"], metadata["group_size"], linear.bias)
                model.set_submodule(module_path, replacement)
                replacements[name] = replacement
                continue
            if specification["storage"] == "int4":
                values = unpack_groups(weights.get_tensor(name + ".packed").numpy(), weights.get_tensor(name + ".scales").numpy(), weights.get_tensor(name + ".offsets").numpy(), specification["shape"], metadata["group_size"])
                tensor = torch.from_numpy(values)
            elif specification["storage"] == "original":
                tensor = weights.get_tensor(name)
            else:
                raise ValueError("Unsupported tensor storage type")
            restored[name] = tensor
    for alias, original in aliases.items():
        if original in replacements:
            model.set_submodule(alias.removesuffix(".weight"), replacements[original])
        else:
            restored[alias] = restored[original]
    model.load_state_dict(restored, strict=True, assign=True)
    model.native_int4_layers = len(replacements)
    model.native_int4_parameters = sum(layer.in_features * layer.out_features for layer in replacements.values())
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--group-size", type=int, choices=[16, 32, 64, 128], default=32)
    args = parser.parse_args()
    build(args.source, args.out, args.group_size, args.license)


if __name__ == "__main__":
    main()
