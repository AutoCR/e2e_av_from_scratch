# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import random, os

import torch
from torch.utils.data._utils.collate import default_collate

def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    # 自定义collate函数处理形状不一致的tensor
    features_batch = []
    targets_batch = []
    tokens_batch = []
    
    for sample in samples_list:
        features, targets, token = sample
        features_batch.append(features)
        targets_batch.append(targets)
        tokens_batch.append(token)
    
    # 尝试对features进行collate，如果失败则保持为列表
    try:
        collated_features = {}
        for key in features_batch[0].keys():
            try:
                # 尝试对每个特征单独进行collate
                collated_features[key] = torch.stack([f[key] for f in features_batch])
            except RuntimeError:
                # 如果失败（形状不一致），则保持为列表
                collated_features[key] = [f[key] for f in features_batch]
    except Exception:
        # 如果整体collate失败，直接使用原始列表
        collated_features = features_batch
    
    # 尝试对targets进行collate，如果失败则保持为列表
    try:
        collated_targets = {}
        for key in targets_batch[0].keys():
            try:
                collated_targets[key] = torch.stack([t[key] for t in targets_batch])
            except RuntimeError:
                collated_targets[key] = [t[key] for t in targets_batch]
    except Exception:
        collated_targets = targets_batch
    
    # 处理mask生成部分
    B = len(samples_list)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)  # [n_samples_masked+1,]
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))  # [np, np]
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)  # len(masks_list) == B

    # upsampling, because vit use AdaptivePooling...
    collated_masks = torch.stack(masks_list)  # [B, np, np]
    collated_masks_up = torch.nn.functional.interpolate(collated_masks.unsqueeze(1).float(), scale_factor=2, mode='nearest').squeeze(1).bool()
    
    if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
        # 可视化和保存masks的代码保持不变
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(collated_masks[0].cpu().numpy())
        plt.title('Original Mask')
        plt.subplot(122)
        plt.imshow(collated_masks_up[0].cpu().numpy())
        plt.title('Upsampled Mask')
        plt.savefig('masks_visualization.png')
        plt.close()

    collated_masks = collated_masks.flatten(1)  # [B, np*np]
    mask_indices_list = collated_masks.flatten().nonzero().flatten()  # [\sum_{i=0}^{B-1}{num_nonzero},]
    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    collated_masks_up = collated_masks_up.flatten(1)

    # 确保collated_features是字典类型再更新
    if isinstance(collated_features, dict):
        collated_features.update({
            "collated_masks": collated_masks_up,
            "mask_indices_list": mask_indices_list,
            "masks_weight": masks_weight,
            "upperbound": upperbound,
            "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
        })

    return collated_features, collated_targets, tokens_batch

def collate_data_and_cast2(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
    #     import pdb; pdb.set_trace()

    # cam_seq_len = 2
    # bs = len(samples_list)

    # collated_ori_teacher_lst = []
    # for i in range(cam_seq_len):
    #     collated_ori_teacher = torch.stack([s[0]["ori_teacher"][i] for s in samples_list])
    #     collated_ori_teacher_lst.append(collated_ori_teacher)
    
    # collated_ori_lst = []
    # for i in range(cam_seq_len):
    #     collated_ori = torch.stack([s[0]["ori"][i] for s in samples_list])
    #     collated_ori_lst.append(collated_ori)

    # num_stu_ensemble = len(samples_list[0][0]["rotated"])
    # collated_rotated_lst = [[] for _ in range(num_stu_ensemble)]
    # for k in range(num_stu_ensemble):
    #     for i in range(cam_seq_len):
    #         collated_rotated = torch.stack([s[0]["rotated"][k][i] for s in samples_list])
    #         collated_rotated_lst[k].append(collated_rotated)

    features, targets, tokens = default_collate(samples_list)

    B = len(features['ori'][0])
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)  # [n_samples_masked+1,]
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))  # [np, np]
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)  # len(masks_list) == n_samples_masked (nsm)

    # upsampling, because vit use AdaptivePooling...
    collated_masks = torch.stack(masks_list)  # [nsm, np, np]
    collated_masks_up = torch.nn.functional.interpolate(collated_masks.unsqueeze(1).float(), scale_factor=2, mode='nearest').squeeze(1).bool()
    
    if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
        # import pdb; pdb.set_trace()
        # Visualize and save masks
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(collated_masks[0].cpu().numpy())
        plt.title('Original Mask')
        plt.subplot(122)
        plt.imshow(collated_masks_up[0].cpu().numpy())
        plt.title('Upsampled Mask')
        plt.savefig('masks_visualization.png')
        plt.close()  # [nsm, 2*np, 2*np]

    collated_masks = collated_masks.flatten(1)  # [nsm, np*np]
    mask_indices_list = collated_masks.flatten().nonzero().flatten()  # [\sum_{i=0}^{bs*n_global_crops-1}{num_nonzero},]
    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    collated_masks_up = collated_masks_up.flatten(1)

    features.update({
        "collated_masks": collated_masks_up,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    })

    return features, targets, tokens

    