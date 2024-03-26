import torch

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    # assert (len(size) == 4)
    other = size[:-2]
    feat_var = feat.reshape(*other, -1).var(dim=-1) + eps
    feat_std = feat_var.sqrt().reshape(*other, 1, 1)
    feat_mean = feat.reshape(*other, -1).mean(dim=-1).reshape(*other, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat = None, is_simplied = False, style_mean = None, style_std = None):
    size = content_feat.size()
    if not is_simplied:
        assert style_feat is not None
        assert (content_feat.size()[:-2] == style_feat.size()[:-2])
        style_mean, style_std = calc_mean_std(style_feat)
    else:
        assert style_mean is not None
        assert style_std is not None
    content_mean, content_std = calc_mean_std(content_feat)
    # return content_feat - content_mean.expand(size) + style_mean.expand(size)
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def block_adaIN(content_feat, style_feat = None, blocknum = 16, is_simplied = False, style_mean = None, style_std = None):
    if not is_simplied:
        assert (content_feat.size()[:-2] == style_feat.size()[:-2])
        content_feat = blockzation(content_feat, blocknum)
        style_feat = blockzation(style_feat, blocknum)
        return  unblockzation(adaptive_instance_normalization(content_feat, style_feat))
    else:
        assert style_mean is not None
        assert style_std is not None
        content_feat = blockzation(content_feat, blocknum)
        return unblockzation(adaptive_instance_normalization(content_feat, is_simplied=True, style_mean=style_mean, style_std=style_std))
        
def frank_wolfe_solver(veclist, ep = 1e-4, maxnum = 20, ind_dim=1):
    shape = veclist.shape
    veclist = veclist.view(*shape[:ind_dim+1], -1) # shape [B, N, O]
    M = veclist @ veclist.transpose(-1,-2) # shape [B, N, N]
    a = (torch.ones(shape[:ind_dim+1]) / shape[ind_dim]).unsqueeze(ind_dim).to(veclist.device) # shape [B, 1, N]
    for _ in range(maxnum):
        minrank = torch.argmin(a @ M, dim = ind_dim+1) # shape [B, 1]
        minonehot = torch.zeros(shape[:ind_dim+1]).to(veclist.device).scatter_(ind_dim, minrank, 1).unsqueeze(ind_dim) # shape [B, 1, N]
        gamma = min_norm_element_from2(minonehot @ M @ minonehot.transpose(-1,-2),minonehot @ M @ a.transpose(-1,-2), a @ M @ a.transpose(-1,-2)).reshape(*shape[:ind_dim], 1, 1)
        # minvec = torch.diagonal(veclist[:,minrank]).transpose(0,1)
        a = (1-gamma)* a + gamma * minonehot
        if torch.abs(gamma).mean()< ep:
            return (a @ veclist).view(*shape[:ind_dim], *shape[ind_dim+1:])
    return (a @ veclist).view(*shape[:ind_dim], *shape[ind_dim+1:])

def blockzation(feat, blocknum = 16):
    H, W = feat.size()[-2:]
    assert H % blocknum == 0
    assert W % blocknum == 0
    size = feat.size()[:-2]
    feat = feat.reshape(*size,blocknum, H // blocknum, blocknum, W // blocknum).transpose(-2, -3)
    return feat

def unblockzation(feat):
    size = feat.size()
    H = size[-4] * size[-2]
    W = size[-3] * size[-1]
    size = size[:-4]
    return feat.transpose(-2, -3).reshape(*size, H, W)

def dynamic_adj_add(vec1, vec2, range = 20):
    # print(vec1.shape)
    assert vec1.shape == vec2.shape
    shape = vec1.shape
    vec1 = vec1.view(shape[0], -1)
    vec2 = vec2.view(shape[0], -1)
    v1v1 = (vec1 * vec1).mean(dim = 1)
    v1v2 = (vec1 * vec2).mean(dim = 1)
    v2v2 = (vec2 * vec2).mean(dim = 1)
    gamma = min_norm_element_from2(v1v1, v1v2, v2v2).view(shape[0], 1)
    # print(gamma)
    coef = ((1 - gamma)/(gamma + 1e-3)).clamp(0,range)
    # print(coef)
    coef = torch.where(torch.isnan(coef), torch.full_like(coef, 0), coef)
    # return (gamma * vec1 + (1 - gamma) * vec2).view(shape)
    return (vec1 + coef * vec2).view(shape)

def min_norm_element_from2(v1v1, v1v2, v2v2):
    divide = v1v1+v2v2 - 2*v1v2
    gamma = -1.0 *  (v1v2 - v2v2) / (divide + 1e-3)
    gamma = torch.where(torch.isnan(gamma), torch.full_like(gamma, 1), gamma)
    return gamma.clamp(0, 1)

# get vertical (to vec_base) componenet of vec 
def get_vertical_component(vec, vec_base, independdims = 1):
    assert vec.shape  == vec_base.shape
    assert vec.device == vec_base.device
    shape = vec.shape
    vec = vec.reshape(*shape[:independdims], -1)
    vec_base = vec_base.reshape(*shape[:independdims], -1)
    cos = (vec * vec_base).sum(dim = -1, keepdim = True) / (vec_base ** 2).sum(dim = -1, keepdim = True)
    vec_align = cos * vec_base
    return (vec - vec_align).reshape(shape)

# get the tangent part of ’delta‘ and vertical part of ‘delta’, and the superball is decide by img and refmean
## technically, tangent part is vertical to 2 vectors and is exactly the corresponding vertical part.
def divide_gradient(img, delta, refmean, blocknum = 16):
    assert img.shape == delta.shape
    assert img.device == refmean.device == delta.device
    blockedimg = blockzation(img, blocknum)
    blockeddelta = blockzation(delta, blocknum)
    meanvec = torch.ones(blockedimg.shape).to(blockedimg.device)
    middledelta = get_vertical_component(blockeddelta, meanvec, -2)
    finaldelta = get_vertical_component(middledelta, blockedimg - refmean, -2)
    resdelta = blockeddelta - finaldelta
    
    return unblockzation(finaldelta), unblockzation(resdelta)

def retraction(img, refmean, refstd, blocknum = 16):
    assert img.device == refmean.device == refstd.device
    block_img = blockzation(img, blocknum)
    block_restractioned = adaptive_instance_normalization(block_img, is_simplied= True, style_mean=refmean, style_std=refstd)

    return unblockzation(block_restractioned)
