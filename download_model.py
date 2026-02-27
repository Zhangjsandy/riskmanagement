# -*- coding: utf-8 -*-
"""
TabPFN模型下载脚本

本脚本用于从Hugging Face Hub下载TabPFN-v2预训练模型文件，
并将其复制到TabPFN的本地缓存目录以便离线使用。

Dependencies:
    - huggingface_hub: 用于从Hugging Face下载模型
    - os, shutil: 用于文件操作


"""

import os
import shutil
from huggingface_hub import hf_hub_download


def download_tabpfn_model():
    """
    下载TabPFN模型文件到本地缓存目录

    该函数执行以下操作：
    1. 从Hugging Face Hub下载TabPFN-v2预训练模型
    2. 将模型文件保存到本地目录
    3. 复制到TabPFN的缓存目录以便离线使用

    Raises:
        Exception: 当下载或文件操作失败时抛出异常
    """
    print("=" * 80)
    print("下载TabPFN模型文件")
    print("=" * 80)

    # 模型信息配置
    repo_id = "Prior-Labs/TabPFN-v2"  # Hugging Face仓库ID
    filename = "tabpfn-v2-classifier-v2_default.ckpt"  # 模型文件名
    local_dir = "./tabpfn-models"  # 本地下载目录

    # 创建本地目录（如果不存在）
    os.makedirs(local_dir, exist_ok=True)
    print(f"\n本地目录: {local_dir}")
    print(f"仓库: {repo_id}")
    print(f"文件名: {filename}")

    # 下载模型
    print("\n开始下载...")
    try:
        # 从Hugging Face Hub下载模型文件
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False  # 不使用符号链接，直接复制文件
        )
        print(f"\n✓ 下载成功!")
        print(f"文件路径: {downloaded_path}")

        # 检查文件大小
        file_size = os.path.getsize(downloaded_path)
        print(f"文件大小: {file_size / (1024*1024):.2f} MB")

        # 复制到TabPFN缓存目录
        cache_dir = os.path.expanduser("~/.cache/tabpfn/models/TabPFN-v2")
        os.makedirs(cache_dir, exist_ok=True)

        # TabPFN期望的文件名（版本2.5）
        target_filename = "tabpfn-v2.5-classifier-v2.5_default.ckpt"
        target_path = os.path.join(cache_dir, target_filename)

        # 复制文件到缓存目录
        shutil.copy2(downloaded_path, target_path)
        print(f"\n✓ 已复制到TabPFN缓存目录: {target_path}")

    except Exception as e:
        # 捕获并显示下载过程中的任何错误
        print(f"\n✗ 下载失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    """脚本入口点，执行模型下载。"""
    download_tabpfn_model()
