import os
import json
import random

def generate_val_json(
    img_dir: str,
    gt_dir: str,
    output_file: str = "val_samples.json",
    ratio: float = 0.1,
    seed: int = 42,
):
    """
    从 img_dir 中随机抽取 ratio 比例的 .npz 文件（最少一个），
    假设 ground-truth npz 在 gt_dir 里同名存在，
    生成一个 JSON 列表，每个元素是 [img_path, gt_path]。

    最终写入 output_file，例如:
    [
      ["path/to/imgs/a.npz", "path/to/gts/a.npz"],
      ["path/to/imgs/b.npz", "path/to/gts/b.npz"],
      ...
    ]
    """
    # 列出所有 .npz 文件
    all_imgs = [f for f in os.listdir(img_dir) if f.endswith(".npz")]
    all_imgs.sort()
    if not all_imgs:
        raise ValueError(f"No .npz files found in {img_dir}")

    # 随机抽样
    random.seed(seed)
    k = max(1, int(len(all_imgs) * ratio))
    sampled = random.sample(all_imgs, k)

    # 生成对应的 (img_path, gt_path) 对
    pairs = []
    for fn in sampled:
        img_path = os.path.join(img_dir, fn)
        gt_path = os.path.join(gt_dir, fn)
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")
        pairs.append([img_path, gt_path])

    # 写入 JSON
    with open(output_file, "w") as f:
        json.dump(pairs, f, indent=4)
    print(f"[generate_val_json] 写入 {len(pairs)} 条样本到 {output_file}")

# 示例调用
if __name__ == "__main__":
    # 根据你的实际路径修改下面两个参数
    img_folder = "/home/sjtu-huang/CVPR2025/3D_val_npz"
    gt_folder  = "/home/sjtu-huang/CVPR2025/3D_val_gt/3D_val_gt_interactive"
    generate_val_json(img_folder, gt_folder,
                      output_file="val_samples.json",
                      ratio=0.2, seed=123)
