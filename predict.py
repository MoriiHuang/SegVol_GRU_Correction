from transformers import AutoTokenizer, AutoConfig
import torch
import os
import numpy as np
from segvol.model_segvol_single import SegVolModel, build_binary_cube_dict, build_binary_points
from tqdm import tqdm
import json
from glob import glob

def load_item(npz_file, preprocessor):
    file_path = npz_file
    npz = np.load(file_path, allow_pickle=True)
    imgs = npz['imgs']
    boxes = npz.get('boxes', [])  # 框提示（可能为空）
    clicks = npz.get('clicks', [])  # 点提示（新增）

    imgs = preprocessor.preprocess_ct_case(imgs)   # (1, H, W, D)
    
    # 处理框提示 -> cube_boxes（原有逻辑）
    cube_boxes = []
    for std_box in boxes:
        binary_cube = build_binary_cube_dict(std_box, imgs.shape[1:])
        cube_boxes.append(binary_cube)
    cube_boxes = torch.stack(cube_boxes, dim=0) if cube_boxes else None
    assert cube_boxes.shape[1:] == imgs.shape[1:], f'{cube_boxes.shape} != {imgs.shape}'
    # 处理点提示 -> point_prompts_map 和 zoom_out_point_prompt（新增）
    point_prompts_map = []
    zoom_out_point_prompt = []
    for click_info in clicks:
        # 解析前景/背景点坐标（格式：[{'fg': [[x,y,z],...], 'bg': [[x,y,z],...]}]）
        fg_points = click_info.get('fg', [])
        bg_points = click_info.get('bg', [])
        if not fg_points and not bg_points:
            point_prompts_map.append(None)
            zoom_out_point_prompt.append(None)
            continue
        
        # 转换为模型需要的点提示格式（坐标+标签）
        points = []
        labels = []
        for p in fg_points:
            points.append(p)
            labels.append(1)  # 前景点标签为1
        for p in bg_points:
            points.append(p)
            labels.append(0)  # 背景点标签为0 or [0]
        points_tensor = torch.tensor(points, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        # 生成点提示和对应的 zoom-out 二值图（使用 processor 的 point_prompt_b 方法）
        # point_prompt, point_prompt_map = preprocessor.point_prompt_b(
        #     points_tensor, labels_tensor, device='cpu'  # 注意设备需与推理时一致
        # )
        #the number of positive points 
        num_positive_points = len(fg_points)
        num_negative_points = len(bg_points)
        point_prompt = (points_tensor.unsqueeze(0).float().cuda(), labels_tensor.unsqueeze(0).float().cuda()) 
        binary_points = build_binary_points(points_tensor, labels_tensor, imgs.shape[1:])
        #print('the shape of binary_points is: ', binary_points.shape)
        spatial_size = (32, 256, 256)
        binary_points_resize = F.interpolate(binary_points.unsqueeze(0).unsqueeze(0).float(),
                    size=spatial_size, mode='nearest')[0][0]         
        point_resize, point_label_resize = select_points(binary_points_resize.squeeze(), num_positive_extra=num_positive_points, num_negative_extra=num_negative_points)
        zoomout_point_prompt = (point_resize.unsqueeze(0).float().cuda(), point_label_resize.unsqueeze(0).float().cuda()) 

        point_prompts_map.append(binary_points)
        zoom_out_point_prompt.append(zoomout_point_prompt)
    # 执行 zoom 变换（原有逻辑扩展）
    zoom_item = preprocessor.zoom_transform_case(imgs, cube_boxes)
    zoom_item.update({
        'file_path': file_path,
        'img_original': torch.from_numpy(npz['imgs']),
        'point_prompts_map': point_prompts_map,  # 新增：点提示列表（每个类别对应一个）
        'zoom_out_point_prompt': zoom_out_point_prompt  # 新增：zoom-out 后的点提示二值图
    })
    return zoom_item

def backfill_foreground_preds(ct_shape, logits_mask, start_coord, end_coord):
    binary_preds = torch.zeros(ct_shape)
    binary_preds[start_coord[0]:end_coord[0], 
                    start_coord[1]:end_coord[1], 
                    start_coord[2]:end_coord[2]] = torch.sigmoid(logits_mask)
    binary_preds = torch.where(binary_preds > 0.5, 1., 0.)
    return binary_preds

def infer_case(model_val, data_item, processor, device):
    data_item['image'], data_item['zoom_out_image'] = \
        data_item['image'].unsqueeze(0).to(device), data_item['zoom_out_image'].unsqueeze(0).to(device)
    start_coord, end_coord = data_item['foreground_start_coord'], data_item['foreground_end_coord']

    img_original = data_item['img_original']
    category_n = data_item['cube_boxes'].shape[0] if data_item['cube_boxes'] is not None else 0
    category_ids = torch.arange(category_n) + 1
    category_ids = list(category_ids)
    final_preds = torch.zeros_like(img_original)

    for category_id in category_ids:
        cls_idx = (category_id - 1).item()
        
        # 处理框提示（原有逻辑）
        if data_item['cube_boxes'] is not None:
            cube_boxes = data_item['cube_boxes'][cls_idx].unsqueeze(0).unsqueeze(0)
            bbox_prompt = processor.bbox_prompt_b(data_item['zoom_out_cube_boxes'][cls_idx], device=device)
            bbox_prompt_group = [bbox_prompt, cube_boxes]
        else:
            bbox_prompt_group = None

        # 处理点提示（新增）
        if data_item['point_prompts_map'] is not None:
            point_prompt_map = data_item['point_prompts_map'][cls_idx] if cls_idx < len(data_item['point_prompts_map']) else None
            point_prompt = data_item['zoom_out_point_prompt'][cls_idx] if cls_idx < len(data_item['zoom_out_point_prompt']) else None
            point_prompt_group = [point_prompt, point_prompt_map] if point_prompt is not None else None
        else:
            point_prompt_group = None

        # 模型推理（同时传递框和点提示）
        with torch.no_grad():
            logits_mask = model_val.forward_test(
                image=data_item['image'],
                zoomed_image=data_item['zoom_out_image'],
                bbox_prompt_group=bbox_prompt_group,  # 框提示组
                point_prompt_group=point_prompt_group,  # 点提示组（新增）
                use_zoom=True
            )
        
        # 后处理（原有逻辑）
        binary_preds = backfill_foreground_preds(img_original.shape, logits_mask, start_coord, end_coord)
        final_preds[binary_preds == 1] = category_id
        
        # 清理缓存（原有逻辑）
        torch.cuda.empty_cache()
    
    return final_preds.numpy()

def main():
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = './segvol'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    ckpt_path = './ckpts_fm3d_segvol_0507/epoch_2025_loss_0.2617.pth'
    clip_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_val = SegVolModel(config)
    model_val.model.text_encoder.tokenizer = clip_tokenizer
    processor = model_val.processor
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_val.load_state_dict(checkpoint['model_state_dict'])
    model_val.eval()
    model_val.to(device)
    
    # 处理输入文件
    out_dir = "./outputs"
    os.makedirs(out_dir, exist_ok=True)

    npz_files = glob("inputs/*.npz")
    for npz_file in npz_files:
        if not npz_file.endswith('.npz'):
            continue
        data_item = load_item(npz_file, processor)
        final_preds = infer_case(model_val, data_item, processor, device)
        output_path = os.path.join(out_dir, os.path.basename(npz_file))
        np.savez_compressed(output_path, segs=final_preds)
    print('done')

if __name__ == "__main__":
    main()