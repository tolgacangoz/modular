#!/usr/bin/env python3
"""Test script to compile Z-Image transformer modules individually.

Run from the modular/max/python directory:
    python -m max.pipelines.architectures.z_image_module_v3.test_compile_modules

This helps identify which specific module causes compilation to hang.
"""

import time
from max.dtype import DType
from max.graph import TensorType, DeviceRef
from max.driver import CPU, scan_available_devices, load_devices


def get_gpu_device():
    """Get the first available GPU device and its DeviceRef."""
    available = scan_available_devices()
    for spec in available:
        if spec.device_type == "gpu":
            devices = load_devices([spec])
            device = devices[0]
            device_ref = DeviceRef(device.label, device.id)
            return device, device_ref
    raise RuntimeError("No GPU device available")


def test_rope_embedder():
    """Test compiling RopeEmbedder alone."""
    print("\n" + "="*60)
    print("Testing RopeEmbedder compilation...")
    print("="*60)

    from max.pipelines.architectures.z_image_module_v3.nn.transformer_z_image import RopeEmbedder

    device, device_ref = get_gpu_device()

    # Create RopeEmbedder with Z-Image default params and GPU device
    rope = RopeEmbedder(
        theta=256.0,
        axes_dims=(32, 48, 48),  # sum = 128 = head_dim
        axes_lens=(1024, 512, 512),
        device=device,  # Pass GPU device for freqs_cis precomputation
    )

    # Input: position IDs of shape (seq_len, 3)
    seq_len = 4096  # image tokens
    ids_type = TensorType(DType.int32, shape=(seq_len, 3), device=device_ref)

    print(f"  Input type: {ids_type}")
    print("  Starting compilation...")
    start = time.perf_counter()

    try:
        compiled = rope.compile(ids_type)
        elapsed = time.perf_counter() - start
        print(f"  SUCCESS! Compilation took {elapsed:.2f}s")
        return True
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"  FAILED after {elapsed:.2f}s: {e}")
        return False


def test_attention():
    """Test compiling ZImageSingleStreamAttention alone."""
    print("\n" + "="*60)
    print("Testing ZImageSingleStreamAttention compilation...")
    print("="*60)

    from max.pipelines.architectures.z_image_module_v3.nn.transformer_z_image import (
        ZImageSingleStreamAttention
    )

    device, device_ref = get_gpu_device()

    dim = 3840
    heads = 30
    head_dim = dim // heads  # 128

    attn = ZImageSingleStreamAttention(
        dim=dim,
        heads=heads,
        dim_head=head_dim,
        eps=1e-5,
    )

    # Input: hidden_states (B, S, C), freqs_cis (B, S, head_dim//2, 2)
    B, S, C = 1, 4096, dim
    hidden_states_type = TensorType(DType.bfloat16, shape=(B, S, C), device=device_ref)
    freqs_cis_type = TensorType(DType.float32, shape=(B, S, head_dim // 2, 2), device=device_ref)

    print(f"  hidden_states type: {hidden_states_type}")
    print(f"  freqs_cis type: {freqs_cis_type}")
    print("  Starting compilation...")
    start = time.perf_counter()

    try:
        # NOTE: Attention with optional freqs_cis is tricky to test standalone
        # because we can't pass None to compile(). For now, compile with just
        # hidden_states (freqs_cis is optional in the signature).
        # The full transformer test is more meaningful.
        compiled = attn.compile(
            hidden_states_type,
            # Skip optional args - compile() can't handle None
        )
        elapsed = time.perf_counter() - start
        print(f"  SUCCESS! Compilation took {elapsed:.2f}s")
        return True
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"  FAILED after {elapsed:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transformer_block():
    """Test compiling ZImageTransformerBlock alone."""
    print("\n" + "="*60)
    print("Testing ZImageTransformerBlock compilation...")
    print("="*60)

    from max.pipelines.architectures.z_image_module_v3.nn.transformer_z_image import (
        ZImageTransformerBlock
    )

    device, device_ref = get_gpu_device()

    dim = 3840
    n_heads = 30
    head_dim = dim // n_heads

    block = ZImageTransformerBlock(
        layer_id=0,
        dim=dim,
        n_heads=n_heads,
        norm_eps=1e-5,
        qk_norm=True,
        modulation=True,
    )

    # Inputs: x (B, S, C), valid_length (None), freqs_cis (B, S, D/2, 2), adaln_input (B, adaln_dim)
    B, S, C = 1, 4096, dim
    adaln_dim = 256  # ADALN_EMBED_DIM

    x_type = TensorType(DType.bfloat16, shape=(B, S, C), device=device_ref)
    freqs_cis_type = TensorType(DType.float32, shape=(B, S, head_dim // 2, 2), device=device_ref)
    adaln_type = TensorType(DType.bfloat16, shape=(B, adaln_dim), device=device_ref)

    print(f"  x type: {x_type}")
    print(f"  freqs_cis type: {freqs_cis_type}")
    print(f"  adaln type: {adaln_type}")
    print("  Starting compilation...")
    start = time.perf_counter()

    try:
        # NOTE: compile() can't handle None args, so we only pass required args.
        # The full transformer test is more meaningful for the complete flow.
        compiled = block.compile(
            x_type,
            # Skip optional args
        )
        elapsed = time.perf_counter() - start
        print(f"  SUCCESS! Compilation took {elapsed:.2f}s")
        return True
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"  FAILED after {elapsed:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_transformer():
    """Test compiling the full ZImageTransformer2DModel."""
    print("\n" + "="*60)
    print("Testing ZImageTransformer2DModel compilation...")
    print("="*60)

    from max.pipelines.architectures.z_image_module_v3.nn.transformer_z_image import (
        ZImageTransformer2DModel
    )

    device, device_ref = get_gpu_device()

    transformer = ZImageTransformer2DModel(
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1024, 512, 512],
        device=device,  # Pass GPU device for RoPE precomputation
    )

    # Same inputs as in model.py compile call
    C, F_dim, H_dim, W_dim = 16, 1, 128, 128
    cap_seq_len = 101

    hidden_states_type = TensorType(DType.bfloat16, shape=(C, F_dim, H_dim, W_dim), device=device_ref)
    t_type = TensorType(DType.float32, shape=(1,), device=device_ref)
    cap_feats_type = TensorType(DType.bfloat16, shape=(cap_seq_len, 2560), device=device_ref)

    print(f"  hidden_states type: {hidden_states_type}")
    print(f"  t type: {t_type}")
    print(f"  cap_feats type: {cap_feats_type}")
    print("  Starting compilation...")
    start = time.perf_counter()

    try:
        compiled = transformer.compile(
            hidden_states_type,
            t_type,
            cap_feats_type,
        )
        elapsed = time.perf_counter() - start
        print(f"  SUCCESS! Compilation took {elapsed:.2f}s")
        return True
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"  FAILED after {elapsed:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("Z-Image Transformer Module Compilation Test")
    print("=" * 60)
    print("This script tests compiling each module individually")
    print("to identify which one causes compilation to hang.")
    print()
    print("Press Ctrl+C to abort if a test hangs.")
    print()

    results = {}

    # Test in order of complexity
    tests = [
        ("RopeEmbedder", test_rope_embedder),
        ("Attention", test_attention),
        ("TransformerBlock", test_transformer_block),
        ("FullTransformer", test_full_transformer),
    ]

    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except KeyboardInterrupt:
            print(f"\n\n*** {name} test interrupted by user - likely hanging! ***")
            results[name] = "HANG"
            break

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, result in results.items():
        status = "PASS" if result == True else ("HANG" if result == "HANG" else "FAIL")
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
