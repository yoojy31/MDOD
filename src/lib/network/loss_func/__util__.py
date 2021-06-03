import torch


def create_target_sim_matrix(box_nums):
    zero_idxes = [[], []]
    for r in range(1, box_nums.shape[0]):
        for c in range(r):
            if box_nums[r] != box_nums[c]:
                zero_idxes[0] += [r, c]
                zero_idxes[1] += [c, r]

    target_sim_mat = torch.ones((box_nums.shape[0], box_nums.shape[0])).cuda()
    if len(zero_idxes) > 0:
        target_sim_mat[zero_idxes] = 0
    return target_sim_mat
