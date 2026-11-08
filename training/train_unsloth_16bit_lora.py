#!/usr/bin/env python3
"""Unsloth LoRA fine-tuning on a full-precision (bf16/fp16) base.

All options: python train_unsloth_16bit_lora.py --help (see train_unsloth_16bit.py).
"""
import os
import sys

if '--unsloth-compile-disable' in sys.argv:
    os.environ['UNSLOTH_COMPILE_DISABLE'] = '1'

from train_unsloth_16bit import main  # noqa: E402

if __name__ == '__main__':
    main(adapter='lora')
