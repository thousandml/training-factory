"""Customer/local installer for a prebuilt CQ distribution tree.

This setup.py does not compile proprietary code. A customer bundle must already
contain these two compiled files inside cq/:
  runtime_ops.<abi>.so
  cq_kernels.<abi>.so

Use ``pip install .``. GGUF is not supported.
"""
from pathlib import Path
from setuptools import find_packages, setup

HERE = Path(__file__).resolve().parent
PKG = HERE / "cq"

runtime_bins = list(PKG.glob("runtime_ops*.so"))
kernel_bins = list(PKG.glob("cq_kernels*.so"))
if not runtime_bins or not kernel_bins:
    raise RuntimeError(
        "This is the customer installer and requires prebuilt private binaries "
        "inside cq/: runtime_ops*.so and cq_kernels*.so. "
        "Build privately with setup_closed.py, then create the customer bundle."
    )

setup(
    name="cq-quant",
    version="0.4.1",
    packages=find_packages(),
    package_data={"cq": ["runtime_ops*.so", "cq_kernels*.so"]},
    include_package_data=True,
    install_requires=["vllm==0.27.1", "safetensors", "huggingface_hub"],
    python_requires=">=3.12,<3.13",
    entry_points={
        "vllm.general_plugins": ["register_cq = cq:register"],
    },
    zip_safe=False,
)
