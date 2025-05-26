import numpy as np
import torch.nn.functional as F
import math
import torch.nn as nn
import torch
from torch import optim
from torch.autograd import Variable

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        # self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            # attn = attn * mask

        attn = F.softmax(attn, dim=-1)
        # attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        # x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d__vae, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d__vae, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d__vae, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d__vae, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d__vae, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d__vae, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # q = self.dropout(self.fc(q))
        q = self.fc(q)
        q += residual

        q = self.layer_norm(q)

        return q, attn

# default tensorflow initialization of linear layers
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


@torch.jit.script
def fused_mean_variance(x, weight):
    mean = torch.sum(x*weight, dim=2, keepdim=True)
    var = torch.sum(weight * (x - mean)**2, dim=2, keepdim=True)
    return mean, var

class IBRNet(nn.Module):
    def __init__(self, in_feat_ch=32, n_samples=64, **kwargs):
        super(IBRNet, self).__init__()
        # self.args = args
        self.anti_alias_pooling = False
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ELU(inplace=True)
        self.n_samples = n_samples
        self.ray_dir_fc = nn.Sequential(nn.Linear(4, 16),
                                        activation_func,
                                        nn.Linear(16, in_feat_ch + 3),
                                        activation_func)

        self.base_fc = nn.Sequential(nn.Linear((in_feat_ch+3)*3, 64),
                                     activation_func,
                                     nn.Linear(64, 32),
                                     activation_func)

        self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                    activation_func,
                                    nn.Linear(32, 33),
                                    activation_func,
                                    )

        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                     activation_func,
                                     nn.Linear(32, 1),
                                     nn.Sigmoid()
                                     )

        self.geometry_fc = nn.Sequential(nn.Linear(32*2+1, 64),
                                         activation_func,
                                         nn.Linear(64, 16),
                                         activation_func)

        self.ray_attention = MultiHeadAttention(4, 16, 4, 4)
        self.out_geometry_fc = nn.Sequential(nn.Linear(16, 16),
                                             activation_func,
                                             nn.Linear(16, 1),
                                             nn.ReLU())

        self.rgb_fc = nn.Sequential(nn.Linear(32+1+4, 16),
                                    activation_func,
                                    nn.Linear(16, 8),
                                    activation_func,
                                    nn.Linear(8, 1))

        self.pos_encoding = self.posenc(d_hid=16, n_samples=self.n_samples)

        self.base_fc.apply(weights_init)
        self.vis_fc2.apply(weights_init)
        self.vis_fc.apply(weights_init)
        self.geometry_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)

    def posenc(self, d_hid, n_samples):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).to("cuda:{}".format(self.args.local_rank)).float().unsqueeze(0)
        return sinusoid_table

    def forward(self, rgb_feat, ray_diff, mask):
        '''
        :param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat] n_views = 1
        :param ray_diff: ray direction difference [n_rays, n_samples, n_views, 4], first 3 channels are directions,
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        '''  
        num_views = rgb_feat.shape[2]
        direction_feat = self.ray_dir_fc(ray_diff)
        rgb_in = rgb_feat[..., :3]
        rgb_feat = rgb_feat + direction_feat
        if self.anti_alias_pooling:
            _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
            exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
            weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=2, keepdim=True)[0]) * mask
            weight = weight / (torch.sum(weight, dim=2, keepdim=True) + 1e-8) # means it will trust the one more with more consistent view point
        else:
            weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)

        # compute mean and variance across different views for each point
        mean, var = fused_mean_variance(rgb_feat, weight)  # [n_rays, n_samples, 1, n_feat]
        globalfeat = torch.cat([mean, var], dim=-1)  # [n_rays, n_samples, 1, 2*n_feat]

        x = torch.cat([globalfeat.expand(-1, -1, num_views, -1), rgb_feat], dim=-1)  # [n_rays, n_samples, n_views, 3*n_feat]
        x = self.base_fc(x)

        x_vis = self.vis_fc(x * weight)
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1]-1, 1], dim=-1)
        vis = F.sigmoid(vis) * mask
        x = x + x_res
        vis = self.vis_fc2(x * vis) * mask
        weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)

        mean, var = fused_mean_variance(x, weight)
        globalfeat = torch.cat([mean.squeeze(2), var.squeeze(2), weight.mean(dim=2)], dim=-1)  # [n_rays, n_samples, 32*2+1]
        globalfeat = self.geometry_fc(globalfeat)  # [n_rays, n_samples, 16]
        # print(globalfeat.shape)
        num_valid_obs = torch.sum(mask, dim=2)
        globalfeat = globalfeat + self.pos_encoding
        # print(globalfeat.shape)
        globalfeat, _ = self.ray_attention(globalfeat, globalfeat, globalfeat,
                                           mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
        sigma = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
        sigma_out = sigma.masked_fill(num_valid_obs < 1, 0.)  # set the sigma of invalid point to zero

        # rgb computation
        x = torch.cat([x, vis, ray_diff], dim=-1)
        x = self.rgb_fc(x)
        x = x.masked_fill(mask == 0, -1e9)
        blending_weights_valid = F.softmax(x, dim=2)  # color blending
        rgb_out = torch.sum(rgb_in*blending_weights_valid, dim=2)
        out = torch.cat([rgb_out, sigma_out], dim=-1)
        return out

class OPONeRF(nn.Module):
    def __init__(self, neuray_in_dim=32, in_feat_ch=32, n_samples=64, **kwargs):
        super().__init__()
        # self.args = args
        self.anti_alias_pooling = False
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ELU(inplace=True)
        self.n_samples = n_samples
        self.ray_dir_fc = nn.Sequential(nn.Linear(4, 16),
                                        activation_func,
                                        nn.Linear(16, in_feat_ch + 3),
                                        activation_func)

        self.base_fc = nn.Sequential(nn.Linear((in_feat_ch+3)*5+neuray_in_dim, 64),
                                     activation_func,
                                     nn.Linear(64, 32),
                                     activation_func)

        self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                    activation_func,
                                    nn.Linear(32, 33),
                                    activation_func,
                                    )

        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                     activation_func,
                                     nn.Linear(32, 1),
                                     nn.Sigmoid()
                                     )

        self.geometry_fc = nn.Sequential(nn.Linear(32*2+1, 64),
                                         activation_func,
                                         nn.Linear(64, 16),
                                         activation_func)

        self.ray_attention = MultiHeadAttention(4, 16, 4, 4)
        self.out_geometry_fc = nn.Sequential(nn.Linear(16, 16),
                                             activation_func,
                                             nn.Linear(16, 1),
                                             nn.ReLU())

        self.rgb_fc = nn.Sequential(nn.Linear(32+1+4, 16),
                                    activation_func,
                                    nn.Linear(16, 8),
                                    activation_func,
                                    nn.Linear(8, 1))

        self.neuray_fc = nn.Sequential(
            nn.Linear(neuray_in_dim, 8,),
            activation_func,
            nn.Linear(8, 1),
        )

        self.pos_encoding = self.posenc(d_hid=16, n_samples=self.n_samples)

        self.base_fc.apply(weights_init)
        self.vis_fc2.apply(weights_init)
        self.vis_fc.apply(weights_init)
        self.geometry_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)
        self.neuray_fc.apply(weights_init)

    def change_pos_encoding(self,n_samples):
        self.pos_encoding = self.posenc(16, n_samples=n_samples)

    def posenc(self, d_hid, n_samples):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).to("cuda:{}".format(0)).float().unsqueeze(0)
        return sinusoid_table

    def forward(self, rgb_feat, neuray_feat, ray_diff, mask):
        '''
        :param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        :param neuray_feat: [n_rays, n_samples, n_views, 32]
        :param ray_diff: ray direction difference [n_rays, n_samples, n_views, 4], first 3 channels are directions,
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        '''
        #print(rgb_feat.shape, 'rgb_feat', neuray_feat.shape, 'neuray_feat')
        num_views = rgb_feat.shape[2]
        direction_feat = self.ray_dir_fc(ray_diff)
        rgb_in = rgb_feat[..., :3]
        rgb_feat = rgb_feat + direction_feat
        if self.anti_alias_pooling:
            _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
            exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
            weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=2, keepdim=True)[0]) * mask
            weight = weight / (torch.sum(weight, dim=2, keepdim=True) + 1e-8) # means it will trust the one more with more consistent view point
        else:
            weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)
            
        # save rgb_feat neuray_feat img_feats as npy
        # rgb_feat_ori = np.load("rgb_feat.npy")
        # neuray_feat_ori = np.load("neuray_feat.npy")
       
        # if rgb_feat.shape[0] / rgb_feat_ori.shape[0] == 8:

        #     rgb_feat_ori = torch.from_numpy(rgb_feat_ori).to("cuda:{}".format(0)).float()
        #     neuray_feat_ori = torch.from_numpy(neuray_feat_ori).to("cuda:{}".format(0)).float()
            
        #     rgb_feat_ori = rgb_feat_ori.repeat(8,1,1,1)
        #     neuray_feat_ori = neuray_feat_ori.repeat(8,1,1,1)
            
        #     mean_feat_ori = torch.mean(rgb_feat_ori, dim=-1)
        #     mean_feat = torch.mean(rgb_feat, dim=-1)
        #     mean_feat = mean_feat.unsqueeze(-1).repeat(1,1,1,rgb_feat.shape[-1])
        #     mean_feat_ori = mean_feat_ori.unsqueeze(-1).repeat(1,1,1,rgb_feat.shape[-1])
        #     var_feat_ori = torch.var(rgb_feat_ori, dim=-1)
        #     var_feat = torch.var(rgb_feat, dim=-1)
        #     var_feat = var_feat.unsqueeze(-1).repeat(1,1,1,rgb_feat.shape[-1])
        #     var_feat_ori = var_feat_ori.unsqueeze(-1).repeat(1,1,1,rgb_feat.shape[-1])
        #     rgb_feat = (rgb_feat - mean_feat)/torch.sqrt(var_feat+1e-8) * torch.sqrt(var_feat_ori+1e-8) + mean_feat_ori
            
        #     mean_neuray_feat_ori = torch.mean(neuray_feat_ori, dim=-1)
        #     mean_neuray_feat = torch.mean(neuray_feat, dim=-1)
        #     mean_neuray_feat = mean_neuray_feat.unsqueeze(-1).repeat(1,1,1,neuray_feat.shape[-1])
        #     mean_neuray_feat_ori = mean_neuray_feat_ori.unsqueeze(-1).repeat(1,1,1,neuray_feat.shape[-1])
        #     var_neuray_feat_ori = torch.var(neuray_feat_ori, dim=-1)
        #     var_neuray_feat = torch.var(neuray_feat, dim=-1)
        #     var_neuray_feat = var_neuray_feat.unsqueeze(-1).repeat(1,1,1,neuray_feat.shape[-1])
        #     var_neuray_feat_ori = var_neuray_feat_ori.unsqueeze(-1).repeat(1,1,1,neuray_feat.shape[-1])
        #     neuray_feat = (neuray_feat - mean_neuray_feat)/torch.sqrt(var_neuray_feat+1e-8) * torch.sqrt(var_neuray_feat_ori+1e-8) + mean_neuray_feat_ori
        

        # neuray layer 0
        weight0 = torch.sigmoid(self.neuray_fc(neuray_feat)) * weight # [rn,dn,rfn,f]
        mean0, var0 = fused_mean_variance(rgb_feat, weight0)  # [n_rays, n_samples, 1, n_feat]
        mean1, var1 = fused_mean_variance(rgb_feat, weight)  # [n_rays, n_samples, 1, n_feat]
        globalfeat = torch.cat([mean0, var0, mean1, var1], dim=-1)  # [n_rays, n_samples, 1, 2*n_feat]

        x = torch.cat([globalfeat.expand(-1, -1, num_views, -1), rgb_feat, neuray_feat], dim=-1)  # [n_rays, n_samples, n_views, 3*n_feat]
        x = self.base_fc(x)

        x_vis = self.vis_fc(x * weight)
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1]-1, 1], dim=-1)
        vis = F.sigmoid(vis) * mask
        x = x + x_res
        vis = self.vis_fc2(x * vis) * mask
        weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)

        mean, var = fused_mean_variance(x, weight) # eliminate the num_view(rfn) dimension here
        globalfeat = torch.cat([mean.squeeze(2), var.squeeze(2), weight.mean(dim=2)], dim=-1)  # [n_rays, n_samples, 32*2+1]
        globalfeat = self.geometry_fc(globalfeat)  # [n_rays, n_samples, 16]
        # print(globalfeat.shape)
        num_valid_obs = torch.sum(mask, dim=2)
        globalfeat = globalfeat + self.pos_encoding
        # print(globalfeat.shape)
        globalfeat, _ = self.ray_attention(globalfeat, globalfeat, globalfeat,
                                           mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
        # print(globalfeat.shape)
        sigma = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
        sigma_out = sigma.masked_fill(num_valid_obs < 1, 0.)  # set the sigma of invalid point to zero

        # rgb computation
        x = torch.cat([x, vis, ray_diff], dim=-1)
        x = self.rgb_fc(x)
        x = x.masked_fill(mask == 0, -1e9)
        blending_weights_valid = F.softmax(x, dim=2)  # color blending
        rgb_out = torch.sum(rgb_in*blending_weights_valid, dim=2)
        out = torch.cat([rgb_out, sigma_out], dim=-1)
        return out

class BinaryStep(torch.autograd.Function):
    @staticmethod 
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input>0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero_index = torch.abs(input) > 1
        middle_index = (torch.abs(input) <= 1) * (torch.abs(input) > 0.4)
        additional = 2-4*torch.abs(input)
        additional[zero_index] = 0.
        additional[middle_index] = 0.4
        return grad_input*additional

class MaskedMLP(nn.Module):
    def __init__(self, in_size, out_size):
        super(MaskedMLP, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.Tensor(out_size, in_size))
        self.bias = nn.Parameter(torch.Tensor(out_size))
        self.threshold = nn.Parameter(torch.Tensor(out_size, in_size))
        self.step = BinaryStep.apply
        #self.mask = None
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound) 
        with torch.no_grad():
            #std = self.weight.std()
            # nn.init.kaiming_uniform_(self.threshold, a=math.sqrt(5))
            self.threshold.data.fill_(0.)
    
    def forward(self, input, weight_matrix, hard_mask):
        abs_weight = torch.abs(self.weight)
        # print(abs_weight)
        # print(self.threshold)
        threshold = self.threshold.view(abs_weight.shape[0], -1)
        abs_weight = abs_weight-threshold
        mask = self.step(abs_weight)
        ratio = torch.sum(mask) / mask.numel()
        #print("keep ratio {:.2f}".format(ratio))
        if ratio <= 0.01:
            with torch.no_grad():
                #std = self.weight.std()
                # self.threshold.data.fill_(0.)
                nn.init.kaiming_uniform_(self.threshold, a=math.sqrt(5))
            abs_weight = torch.abs(self.weight)
            threshold = self.threshold.view(abs_weight.shape[0], -1)
            abs_weight = abs_weight-threshold
            mask = self.step(abs_weight)
        # print(mask.shape,mask)
        # 0 noplus 1 plus
        # print(self.weight.shape,mask.shape,weight_matrix.shape)
        masked_weight = self.weight*mask + weight_matrix.transpose(0,1) * (1-mask)
        # print(input.shape,masked_weight.shape)
        output1 = torch.nn.functional.linear(input, masked_weight, self.bias)
        output2 = torch.nn.functional.linear(input, self.weight, self.bias)
        # print(hard_mask.shape, output1.shape, output2.shape)
        return output1*hard_mask.unsqueeze(dim=-1) + output2*(1-hard_mask).unsqueeze(dim=-1)

class OPONeRFandMixLinear(nn.Module):
    def __init__(self, neuray_in_dim=32, in_feat_ch=32, n_samples=64, **kwargs):
        super().__init__()
        # self.args = args
        self.anti_alias_pooling = False
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ELU(inplace=True)
        self.n_samples = n_samples
        self.ray_dir_fc = nn.Sequential(nn.Linear(4, 16),
                                        activation_func,
                                        nn.Linear(16, in_feat_ch + 3),
                                        activation_func)

        self.base_fc = nn.Sequential(nn.Linear((in_feat_ch+3)*5+neuray_in_dim, 64), # 207,64
                                     activation_func,
                                     nn.Linear(64, 32),
                                     activation_func)
        
        #TODO: hyper_linear here
        self.act_func = nn.ELU(inplace=True)
        self.hyper_linear1 = nn.ModuleList([nn.Sequential(nn.Linear(136*180,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU())] * 2)
        self.hyper_linear2 = nn.ModuleList([nn.Sequential(nn.Linear(32*128,256), nn.ReLU(), nn.Linear(256,512), nn.ReLU())] * 2)
        self.hyper_linear3 = nn.ModuleList([nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 207*64), nn.ReLU()),
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 64*32), nn.ReLU())])
        self.ori_matrix = nn.ParameterList([nn.Parameter(nn.init.kaiming_normal_(torch.zeros([207,64]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([64,32])))])

        self.masked_mlp = nn.ModuleList([MaskedMLP(207, 64),
                                        MaskedMLP(64, 32)])
        self.hard_mask = nn.Sequential(nn.Linear(35,32), activation_func, nn.Linear(32,2),nn.Sigmoid())

        self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                    activation_func,
                                    nn.Linear(32, 33),
                                    activation_func,
                                    )

        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                     activation_func,
                                     nn.Linear(32, 1),
                                     nn.Sigmoid()
                                     )

        self.geometry_fc = nn.Sequential(nn.Linear(32*2+1, 64),
                                         activation_func,
                                         nn.Linear(64, 16),
                                         activation_func)

        self.ray_attention = MultiHeadAttention(4, 16, 4, 4)
        self.out_geometry_fc = nn.Sequential(nn.Linear(16, 16),
                                             activation_func,
                                             nn.Linear(16, 1),
                                             nn.ReLU())

        self.rgb_fc = nn.Sequential(nn.Linear(32+1+4, 16),
                                    activation_func,
                                    nn.Linear(16, 8),
                                    activation_func,
                                    nn.Linear(8, 1))

        self.neuray_fc = nn.Sequential(
            nn.Linear(neuray_in_dim, 8,),
            activation_func,
            nn.Linear(8, 1),
        )

        self.pos_encoding = self.posenc(d_hid=16, n_samples=self.n_samples)

        self.base_fc.apply(weights_init)
        self.vis_fc2.apply(weights_init)
        self.vis_fc.apply(weights_init)
        self.geometry_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)
        self.neuray_fc.apply(weights_init)

    def change_pos_encoding(self,n_samples):
        self.pos_encoding = self.posenc(16, n_samples=n_samples)

    def posenc(self, d_hid, n_samples):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).to("cuda:{}".format(0)).float().unsqueeze(0)
        return sinusoid_table

    def forward(self, rgb_feat, neuray_feat, ray_diff, mask, img_feats, que_imgs_info):
        '''
        :param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        :param neuray_feat: [n_rays, n_samples, n_views, 32]
        :param ray_diff: ray direction difference [n_rays, n_samples, n_views, 4], first 3 channels are directions,
        :param img_feats: [3, 32, 136, 180]
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        '''
        #print(rgb_feat.shape, 'rgb_feat', neuray_feat.shape, 'neuray_feat')
        # print(img_feats.shape, que_imgs_info["imgs"].shape, "!!!!!!")
        num_views = rgb_feat.shape[2]
        direction_feat = self.ray_dir_fc(ray_diff)
        rgb_in = rgb_feat[..., :3]
        rgb_feat = rgb_feat + direction_feat
        if self.anti_alias_pooling:
            _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
            exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
            weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=2, keepdim=True)[0]) * mask
            weight = weight / (torch.sum(weight, dim=2, keepdim=True) + 1e-8) # means it will trust the one more with more consistent view point
        else:
            weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)

        # neuray layer 0
        weight0 = torch.sigmoid(self.neuray_fc(neuray_feat)) * weight # [rn,dn,rfn,f]
        mean0, var0 = fused_mean_variance(rgb_feat, weight0)  # [n_rays, n_samples, 1, n_feat]
        mean1, var1 = fused_mean_variance(rgb_feat, weight)  # [n_rays, n_samples, 1, n_feat]
        globalfeat = torch.cat([mean0, var0, mean1, var1], dim=-1)  # [n_rays, n_samples, 1, 2*n_feat]

        x = torch.cat([globalfeat.expand(-1, -1, num_views, -1), rgb_feat, neuray_feat], dim=-1)  # [n_rays, n_samples, n_views, 3*n_feat]
        
        # TODO:use hyper linear to replace base_fc    
        # x = self.base_fc(x)
        # print(direction_feat.shape)
        hard_mask = self.hard_mask(direction_feat)
        hyper_matrix = []
        for i in range(0,2):
            mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
            mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
            mtx = self.hyper_linear3[i](mtx.view(-1))
            mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
            hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
            # x = self.act_func(torch.matmul(x, hyper_matrix[i]))
            x = self.act_func(self.masked_mlp[i](x, hyper_matrix[i], hard_mask[:,:,:,i]))

        x_vis = self.vis_fc(x * weight)
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1]-1, 1], dim=-1)
        vis = F.sigmoid(vis) * mask
        x = x + x_res
        vis = self.vis_fc2(x * vis) * mask
        weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)

        mean, var = fused_mean_variance(x, weight) # eliminate the num_view(rfn) dimension here
        globalfeat = torch.cat([mean.squeeze(2), var.squeeze(2), weight.mean(dim=2)], dim=-1)  # [n_rays, n_samples, 32*2+1]
        globalfeat = self.geometry_fc(globalfeat)  # [n_rays, n_samples, 16]
        num_valid_obs = torch.sum(mask, dim=2)
        globalfeat = globalfeat + self.pos_encoding
        globalfeat, _ = self.ray_attention(globalfeat, globalfeat, globalfeat,
                                           mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
        sigma = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
        #TODO
        sigma2 = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
        sigma = hard_mask[:,:,1,0].unsqueeze(-1)
        sigma = sigma2 * torch.exp(-sigma*2.5)
        # sigma = torch.mean(hard_mask,dim=[-2,-1],keepdim=False).unsqueeze(-1) 
        # vmax = torch.max(sigma)
        # vmin = torch.min(sigma)
        
        # sigma = (sigma-vmin)/(vmax-vmin)
        # sigma[sigma2<0.1] = 0
        sigma_out = sigma.masked_fill(num_valid_obs < 1, 0.)  # set the sigma of invalid point to zero

        # rgb computation
        x = torch.cat([x, vis, ray_diff], dim=-1)
        x = self.rgb_fc(x)
        x = x.masked_fill(mask == 0, -1e9)
        blending_weights_valid = F.softmax(x, dim=2)  # color blending
        rgb_out = torch.sum(rgb_in*blending_weights_valid, dim=2)
        out = torch.cat([rgb_out, sigma_out], dim=-1)
        # print("here")
        return out

class OPONeRFandMoreMixLinear(nn.Module):
    def __init__(self, neuray_in_dim=32, in_feat_ch=32, n_samples=64, **kwargs):
        super().__init__()
        # self.args = args
        self.anti_alias_pooling = False
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ELU(inplace=True)
        self.n_samples = n_samples
        self.ray_dir_fc = nn.Sequential(nn.Linear(4, 16),
                                        activation_func,
                                        nn.Linear(16, in_feat_ch + 3),
                                        activation_func)

        # self.base_fc = nn.Sequential(nn.Linear((in_feat_ch+3)*5+neuray_in_dim, 64), # 207,64
        #                              activation_func,
        #                              nn.Linear(64, 32),
        #                              activation_func)
        
        #TODO: hyper_linear here
        self.act_func = nn.ELU(inplace=True)
        self.hyper_linear1 = nn.ModuleList([nn.Sequential(nn.Linear(25024,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU())] * 10)
        # self.hyper_linear1 = nn.ModuleList([nn.Sequential(nn.Linear(136*180,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU())] * 10)
        self.hyper_linear2 = nn.ModuleList([nn.Sequential(nn.Linear(32*128,256), nn.ReLU(), nn.Linear(256,512), nn.ReLU())] * 10)
        self.hyper_linear3 = nn.ModuleList([nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*8), nn.ReLU()), #neuray fc
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 207*64), nn.ReLU()), #base fc
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 64*32), nn.ReLU()),
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*32), nn.ReLU()), #vis fc
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*33), nn.ReLU()),
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*32), nn.ReLU()), #vis fc2
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*1), nn.ReLU()),
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 65*64), nn.ReLU()), #geometry fc
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 64*16), nn.ReLU()), 
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 37*16), nn.ReLU()) #rgb fc
                                            ])
        self.ori_matrix = nn.ParameterList([nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,8]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([207,64]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([64,32]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,32]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,33]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,32]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,1]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([65,64]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([64,16]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([37,16])))
                                            ])

        self.masked_mlp = nn.ModuleList([MaskedMLP(32, 8),
                                        MaskedMLP(207, 64),
                                        MaskedMLP(64, 32),
                                        MaskedMLP(32, 32),
                                        MaskedMLP(32, 33),
                                        MaskedMLP(32, 32),
                                        MaskedMLP(32, 1),
                                        MaskedMLP(65, 64),
                                        MaskedMLP(64, 16),
                                        MaskedMLP(37, 16),
                                        ])
        self.hard_mask = nn.Sequential(nn.Linear(35,32), activation_func, nn.Linear(32,8),nn.Sigmoid())
        self.hard_mask2 = nn.Sequential(nn.Linear(105,32), activation_func, nn.Linear(32,2),nn.Sigmoid())

        # self.vis_fc = nn.Sequential(nn.Linear(32, 32),
        #                             activation_func,
        #                             nn.Linear(32, 33),
        #                             activation_func,
        #                             )

        # self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
        #                              activation_func,
        #                              nn.Linear(32, 1),
        #                              nn.Sigmoid()
        #                              )
        self.vis_fc2_sig = nn.Sigmoid()

        self.geometry_fc = nn.Sequential(nn.Linear(32*2+1, 64),
                                         activation_func,
                                         nn.Linear(64, 16),
                                         activation_func)

        self.ray_attention = MultiHeadAttention(4, 16, 4, 4)
        self.out_geometry_fc = nn.Sequential(nn.Linear(16, 16),
                                             activation_func,
                                             nn.Linear(16, 1),
                                             nn.ReLU())

        self.rgb_fc = nn.Sequential(nn.Linear(32+1+4, 16),
                                    activation_func,
                                    nn.Linear(16, 8),
                                    activation_func,
                                    nn.Linear(8, 1))
        # self.rgb_fc = nn.Sequential(nn.Linear(16, 8),
        #                             activation_func,
        #                             nn.Linear(8, 1))

        self.neuray_fc = nn.Sequential(
            nn.Linear(neuray_in_dim, 8,),
            activation_func,
            nn.Linear(8, 1),
        )
        # self.neuray_fc = nn.Sequential(
        #     nn.Linear(8, 1),
        # )

        self.pos_encoding = self.posenc(d_hid=16, n_samples=self.n_samples)

        # self.base_fc.apply(weights_init)
        # self.vis_fc2.apply(weights_init)
        # self.vis_fc.apply(weights_init)
        # self.geometry_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)
        self.neuray_fc.apply(weights_init)
        self.hyper_linear1.apply(weights_init)
        self.hyper_linear2.apply(weights_init)
        self.hyper_linear3.apply(weights_init)
        self.hard_mask.apply(weights_init)
        self.hard_mask2.apply(weights_init)
        

    def change_pos_encoding(self,n_samples):
        self.pos_encoding = self.posenc(16, n_samples=n_samples)

    def posenc(self, d_hid, n_samples):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).to("cuda:{}".format(0)).float().unsqueeze(0)
        return sinusoid_table

    def forward(self, rgb_feat, neuray_feat, ray_diff, mask, img_feats, que_imgs_info):
        '''
        :param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        :param neuray_feat: [n_rays, n_samples, n_views, 32]
        :param ray_diff: ray direction difference [n_rays, n_samples, n_views, 4], first 3 channels are directions,
        :param img_feats: [3, 32, 136, 180]
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        '''
        #print(rgb_feat.shape, 'rgb_feat', neuray_feat.shape, 'neuray_feat')
        # print(img_feats.shape, que_imgs_info["imgs"].shape, "!!!!!!")
        num_views = rgb_feat.shape[2]
        direction_feat = self.ray_dir_fc(ray_diff)
        rgb_in = rgb_feat[..., :3]
        rgb_feat = rgb_feat + direction_feat
        a = rgb_feat.detach()
        
       
        if self.anti_alias_pooling:
            _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
            exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
            weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=2, keepdim=True)[0]) * mask
            weight = weight / (torch.sum(weight, dim=2, keepdim=True) + 1e-8) # means it will trust the one more with more consistent view point
        else:
            weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)

        # neuray layer 0
        # replace neuray_feat
        weight0 = torch.sigmoid(self.neuray_fc(neuray_feat)) * weight # [rn,dn,rfn,f]
        hard_mask = self.hard_mask(direction_feat)
        # hard_mask2 = self.hard_mask2(direction_feat.reshape(direction_feat.shape[0],direction_feat.shape[1],-1))
        hyper_matrix = [torch.zeros(1,1)]
        
        # replace neuray_feat
        mean0, var0 = fused_mean_variance(rgb_feat, weight0)  # [n_rays, n_samples, 1, n_feat]
        mean1, var1 = fused_mean_variance(rgb_feat, weight)  # [n_rays, n_samples, 1, n_feat]
        globalfeat = torch.cat([mean0, var0, mean1, var1], dim=-1)  # [n_rays, n_samples, 1, 2*n_feat]

        x = torch.cat([globalfeat.expand(-1, -1, num_views, -1), rgb_feat, neuray_feat], dim=-1)  # [n_rays, n_samples, n_views, 3*n_feat]
        
        # TODO:use hyper linear to replace base_fc    
        # x = self.base_fc(x)
        # print(direction_feat.shape)
        # print(neuray_feat.shape, x.shape,"nr,x")
        for i in range(1,3):
            mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
            mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
            mtx = self.hyper_linear3[i](mtx.view(-1))
            mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
            hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
            # x = self.act_func(torch.matmul(x, hyper_matrix[i]))
            x = self.act_func(self.masked_mlp[i](x, hyper_matrix[i], hard_mask[:,:,:,i]))

        # replace vis_fc
        # x_vis = self.vis_fc(x * weight)
        x_t = x * weight
        for i in range(3,5):
            mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
            mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
            mtx = self.hyper_linear3[i](mtx.view(-1))
            mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
            hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
            # x = self.act_func(torch.matmul(x, hyper_matrix[i]))
            x_t = self.act_func(self.masked_mlp[i](x_t, hyper_matrix[i], hard_mask[:,:,:,i]))
        x_vis = x_t
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1]-1, 1], dim=-1)
        vis = F.sigmoid(vis) * mask
        x = x + x_res
        # replace vis_fc2
        # vis = self.vis_fc2(x * vis) * mask
        x_t = x * vis 
        # print(x_t.shape, "x_t")
        for i in range(5,7):
            mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
            mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
            mtx = self.hyper_linear3[i](mtx.view(-1))
            mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
            hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
            # x = self.act_func(torch.matmul(x, hyper_matrix[i]))
            if i==4:
                x_t = self.act_func(self.masked_mlp[i](x_t, hyper_matrix[i], hard_mask[:,:,:,i]))
            else:
                x_t = self.vis_fc2_sig(self.masked_mlp[i](x_t, hyper_matrix[i], hard_mask[:,:,:,i]))
        vis = x_t * mask

        weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)

        mean, var = fused_mean_variance(x, weight) # eliminate the num_view(rfn) dimension here
        globalfeat = torch.cat([mean.squeeze(2), var.squeeze(2), weight.mean(dim=2)], dim=-1)  # [n_rays, n_samples, 32*2+1]
        # replace geometry_fc
        globalfeat = self.geometry_fc(globalfeat)  # [n_rays, n_samples, 16]
        # for i in range(7,9):
        #     mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
        #     mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
        #     mtx = self.hyper_linear3[i](mtx.view(-1))
        #     mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
        #     hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
        #     # x = self.act_func(torch.matmul(x, hyper_matrix[i]))
        #     globalfeat = self.act_func(self.masked_mlp[i](globalfeat, hyper_matrix[i], hard_mask2[:,:,(i-7)]))
        # replace geometry_fc
        num_valid_obs = torch.sum(mask, dim=2)
        globalfeat = globalfeat + self.pos_encoding
        globalfeat, _ = self.ray_attention(globalfeat, globalfeat, globalfeat,
                                           mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
        sigma = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
        sigma_out = sigma.masked_fill(num_valid_obs < 1, 0.)  # set the sigma of invalid point to zero

        # rgb computation
        x = torch.cat([x, vis, ray_diff], dim=-1)
        #replace rgb_fc
        x = self.rgb_fc(x)
        # for i in range(9,10):
        #     mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
        #     mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
        #     mtx = self.hyper_linear3[i](mtx.view(-1))
        #     mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
        #     hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
        #     x = self.act_func(self.masked_mlp[i](x, hyper_matrix[i], hard_mask[:,:,:,(i-2)]))
        # x = self.rgb_fc(x)
        #replace rgb_fc
        x = x.masked_fill(mask == 0, -1e9)
        blending_weights_valid = F.softmax(x, dim=2)  # color blending
        rgb_out = torch.sum(rgb_in*blending_weights_valid, dim=2)
        out = torch.cat([rgb_out, sigma_out], dim=-1)
        return out


class OPONeRFand4MixLinear(nn.Module):
    def __init__(self, neuray_in_dim=32, in_feat_ch=32, n_samples=64, **kwargs):
        super().__init__()
        # self.args = args
        self.anti_alias_pooling = False
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ELU(inplace=True)
        self.n_samples = n_samples
        self.ray_dir_fc = nn.Sequential(nn.Linear(4, 16),
                                        activation_func,
                                        nn.Linear(16, in_feat_ch + 3),
                                        activation_func)

        # self.base_fc = nn.Sequential(nn.Linear((in_feat_ch+3)*5+neuray_in_dim, 64), # 207,64
        #                              activation_func,
        #                              nn.Linear(64, 32),
        #                              activation_func)
        
        #TODO: hyper_linear here
        self.act_func = nn.ELU(inplace=True)
        self.hyper_linear1 = nn.ModuleList([nn.Sequential(nn.Linear(25024,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU())] * 10) #vr dataset
        # self.hyper_linear1 = nn.ModuleList([nn.Sequential(nn.Linear(136*180,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU())] * 10)
        self.hyper_linear2 = nn.ModuleList([nn.Sequential(nn.Linear(32*128,256), nn.ReLU(), nn.Linear(256,512), nn.ReLU())] * 10)
        self.hyper_linear3 = nn.ModuleList([nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*8), nn.ReLU()), #neuray fc
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 207*64), nn.ReLU()), #base fc
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 64*32), nn.ReLU()),
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*32), nn.ReLU()), #vis fc
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*33), nn.ReLU()),
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*32), nn.ReLU()), #vis fc2
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*1), nn.ReLU()),
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 65*64), nn.ReLU()), #geometry fc
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 64*16), nn.ReLU()), 
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 37*16), nn.ReLU()) #rgb fc
                                            ])
        self.mu_layer = nn.Sequential(nn.Linear(25024,256), nn.ReLU(), nn.Linear(256, 128),)
        self.sigma_layer = nn.Sequential(nn.Linear(25024,256), nn.ReLU(), nn.Linear(256, 128))
        self.fi_layer = nn.Sequential(nn.Linear(25024,256), nn.ReLU(), nn.Linear(256, 128))
        self.enc = nn.Sequential(nn.Linear(25024,256), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(128,256), nn.ReLU())
        self.residue_layer = nn.Linear(25024,128)
        
        self.ori_matrix = nn.ParameterList([nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,8]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([207,64]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([64,32]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,32]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,33]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,32]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,1]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([65,64]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([64,16]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([37,16])))
                                            ])

        self.masked_mlp = nn.ModuleList([MaskedMLP(32, 8),
                                        MaskedMLP(207, 64),
                                        MaskedMLP(64, 32),
                                        MaskedMLP(32, 32),
                                        MaskedMLP(32, 33),
                                        MaskedMLP(32, 32),
                                        MaskedMLP(32, 1),
                                        MaskedMLP(65, 64),
                                        MaskedMLP(64, 16),
                                        MaskedMLP(37, 16),
                                        ])
        self.hard_mask = nn.Sequential(nn.Linear(35,32), activation_func, nn.Linear(32,8),nn.Sigmoid())
        self.hard_mask2 = nn.Sequential(nn.Linear(105,32), activation_func, nn.Linear(32,2),nn.Sigmoid())
        self.fx2hard_mask = nn.Sequential(nn.Linear(3840000,32), activation_func, nn.Linear(32,786432),nn.Sigmoid())

        # self.vis_fc = nn.Sequential(nn.Linear(32, 32),
        #                             activation_func,
        #                             nn.Linear(32, 33),
        #                             activation_func,
        #                             )

        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                     activation_func,
                                     nn.Linear(32, 1),
                                     nn.Sigmoid()
                                     )
        self.vis_fc2_sig = nn.Sigmoid()

        self.geometry_fc = nn.Sequential(nn.Linear(32*2+1, 64),
                                         activation_func,
                                         nn.Linear(64, 16),
                                         activation_func)

        self.ray_attention = MultiHeadAttention(4, 16, 4, 4)
        self.out_geometry_fc = nn.Sequential(nn.Linear(16, 16),
                                             activation_func,
                                             nn.Linear(16, 1),
                                             nn.ReLU())

        self.rgb_fc = nn.Sequential(nn.Linear(32+1+4, 16),
                                    activation_func,
                                    nn.Linear(16, 8),
                                    activation_func,
                                    nn.Linear(8, 1))
        # self.rgb_fc = nn.Sequential(nn.Linear(16, 8),
        #                             activation_func,
        #                             nn.Linear(8, 1))

        self.neuray_fc = nn.Sequential(
            nn.Linear(neuray_in_dim, 8,),
            activation_func,
            nn.Linear(8, 1),
        )
        # self.neuray_fc = nn.Sequential(
        #     nn.Linear(8, 1),
        # )

        self.pos_encoding = self.posenc(d_hid=16, n_samples=self.n_samples)

        # self.base_fc.apply(weights_init)
        self.vis_fc2.apply(weights_init)
        # self.vis_fc.apply(weights_init)
        self.geometry_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)
        self.neuray_fc.apply(weights_init)
        self.hyper_linear1.apply(weights_init)
        self.hyper_linear2.apply(weights_init)
        self.hyper_linear3.apply(weights_init)
        self.hard_mask.apply(weights_init)
        self.hard_mask2.apply(weights_init)
        
        self.mu_layer.apply(weights_init)
        self.sigma_layer.apply(weights_init)
        self.fi_layer.apply(weights_init)
        self.enc.apply(weights_init)
        self.dec.apply(weights_init)
        self.residue_layer.apply(weights_init)


 
        

    def change_pos_encoding(self,n_samples):
        self.pos_encoding = self.posenc(16, n_samples=n_samples)

    def posenc(self, d_hid, n_samples):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).to("cuda:{}".format(0)).float().unsqueeze(0)
        return sinusoid_table
    
    def prob_learning(self, fx):
        # fx = img_feats.view(img_feats.shape[0],img_feats.shape[1],-1)
        mu = self.mu_layer(fx)
        logvar = self.sigma_layer(fx)
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma)
        fv = mu + eps * sigma
        fi = self.fi_layer(fx)
        Fx = 0.5 * fv + 0.5 * fi
        Fx = Fx + self.residue_layer(fx)
        return Fx, mu, logvar

    def recon_loss(self, x, fx):
        return F.mse_loss(x, fx, reduction='sum')
    
    def KL_loss(self, mu, logvar):
        return 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, rgb_feat, neuray_feat, ray_diff, mask, img_feats, que_imgs_info):
        
        num_views = rgb_feat.shape[2]
        direction_feat = self.ray_dir_fc(ray_diff)
        rgb_in = rgb_feat[..., :3]
        rgb_feat = rgb_feat + direction_feat
        
        
            
        if self.anti_alias_pooling:
            _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
            exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
            weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=2, keepdim=True)[0]) * mask
            weight = weight / (torch.sum(weight, dim=2, keepdim=True) + 1e-8) # means it will trust the one more with more consistent view point
        else:
            weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)

        # neuray layer 0
        # replace neuray_feat
        weight0 = torch.sigmoid(self.neuray_fc(neuray_feat)) * weight # [rn,dn,rfn,f]
        hard_mask = self.hard_mask(direction_feat)
        
        # make hard mask as random mask
        # hard_mask = torch.rand(hard_mask.shape).to("cuda:{}".format(0))
        
        hyper_matrix = [torch.zeros(1,1)]
        
        mean0, var0 = fused_mean_variance(rgb_feat, weight0)  # [n_rays, n_samples, 1, n_feat]
        mean1, var1 = fused_mean_variance(rgb_feat, weight)  # [n_rays, n_samples, 1, n_feat]
        
        globalfeat = torch.cat([mean0, var0, mean1, var1], dim=-1)  # [n_rays, n_samples, 1, 2*n_feat]
        x = torch.cat([globalfeat.expand(-1, -1, num_views, -1), rgb_feat, neuray_feat], dim=-1)  # [n_rays, n_samples, n_views, 3*n_feat]
       
        
        
        # print(img_feats.view(img_feats.shape[0],img_feats.shape[1],-1).shape)
        
        for i in range(1,3):
            fx = img_feats.view(img_feats.shape[0],img_feats.shape[1],-1)
            mtx = self.hyper_linear1[i](fx) # simplified
            mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
            mtx = self.hyper_linear3[i](mtx.view(-1))
            mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
            hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
            # x = self.act_func(torch.matmul(x, hyper_matrix[i]))
            x = self.act_func(self.masked_mlp[i](x, hyper_matrix[i], hard_mask[:,:,:,i]))
        # assert 1==0
        # replace vis_fc
        # x_vis = self.vis_fc(x * weight)
        x_t = x * weight
        for i in range(3,5):
            fx = img_feats.view(img_feats.shape[0],img_feats.shape[1],-1)
            mtx = self.hyper_linear1[i](fx)
            mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
            mtx = self.hyper_linear3[i](mtx.view(-1))
            mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
            hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
            # x = self.act_func(torch.matmul(x, hyper_matrix[i]))
            x_t = self.act_func(self.masked_mlp[i](x_t, hyper_matrix[i], hard_mask[:,:,:,i]))
            
        
        x_vis = x_t
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1]-1, 1], dim=-1)
        vis = F.sigmoid(vis) * mask
        x = x + x_res
        # replace vis_fc2
        vis = self.vis_fc2(x * vis) * mask
       

        weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)

        mean, var = fused_mean_variance(x, weight) # eliminate the num_view(rfn) dimension here
        globalfeat = torch.cat([mean.squeeze(2), var.squeeze(2), weight.mean(dim=2)], dim=-1)  # [n_rays, n_samples, 32*2+1]
        # replace geometry_fc
        globalfeat = self.geometry_fc(globalfeat)  # [n_rays, n_samples, 16]
        
       
        num_valid_obs = torch.sum(mask, dim=2)
        globalfeat = globalfeat + self.pos_encoding
        globalfeat, _ = self.ray_attention(globalfeat, globalfeat, globalfeat,
                                           mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
        sigma = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
        sigma_out = sigma.masked_fill(num_valid_obs < 1, 0.)  # set the sigma of invalid point to zero

        # rgb computation
        x = torch.cat([x, vis, ray_diff], dim=-1)
        #replace rgb_fc
        x = self.rgb_fc(x)
       
        x = x.masked_fill(mask == 0, -1e9)
        blending_weights_valid = F.softmax(x, dim=2)  # color blending
        rgb_out = torch.sum(rgb_in*blending_weights_valid, dim=2)
        out = torch.cat([rgb_out, sigma_out], dim=-1)
        
        # Fx = Fx_final/4
        # mu = mu_final/4
        # logvar = logvar_final/4
        # recon_loss = self.recon_loss(self.enc(fx), self.dec(Fx))
        # kl_loss = self.KL_loss(mu, logvar)
        # print(out.shape)
        return out

class OPONeRFand2MixLinear(nn.Module):
    def __init__(self, neuray_in_dim=32, in_feat_ch=32, n_samples=64, **kwargs):
        super().__init__()
        # self.args = args
        self.anti_alias_pooling = False
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ELU(inplace=True)
        self.n_samples = n_samples
        self.ray_dir_fc = nn.Sequential(nn.Linear(4, 16),
                                        activation_func,
                                        nn.Linear(16, in_feat_ch + 3),
                                        activation_func)

        # self.base_fc = nn.Sequential(nn.Linear((in_feat_ch+3)*5+neuray_in_dim, 64), # 207,64
        #                              activation_func,
        #                              nn.Linear(64, 32),
        #                              activation_func)
        
        #TODO: hyper_linear here
        self.act_func = nn.ELU(inplace=True)
        self.hyper_linear1 = nn.ModuleList([nn.Sequential(nn.Linear(25024,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU())] * 10)
        # self.hyper_linear1 = nn.ModuleList([nn.Sequential(nn.Linear(200*200,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU())] * 10)
        # self.hyper_linear1 = nn.ModuleList([nn.Sequential(nn.Linear(136*184,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU())] * 10)
        self.hyper_linear2 = nn.ModuleList([nn.Sequential(nn.Linear(32*128,256), nn.ReLU(), nn.Linear(256,512), nn.ReLU())] * 10)
        self.hyper_linear3 = nn.ModuleList([nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*8), nn.ReLU()), #neuray fc
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 207*64), nn.ReLU()), #base fc
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 64*32), nn.ReLU()),
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*32), nn.ReLU()), #vis fc
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*33), nn.ReLU()),
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*32), nn.ReLU()), #vis fc2
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*1), nn.ReLU()),
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 65*64), nn.ReLU()), #geometry fc
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 64*16), nn.ReLU()), 
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 37*16), nn.ReLU()) #rgb fc
                                            ])
        self.ori_matrix = nn.ParameterList([nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,8]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([207,64]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([64,32]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,32]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,33]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,32]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,1]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([65,64]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([64,16]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([37,16])))
                                            ])

        self.masked_mlp = nn.ModuleList([MaskedMLP(32, 8),
                                        MaskedMLP(207, 64),
                                        MaskedMLP(64, 32),
                                        MaskedMLP(32, 32),
                                        MaskedMLP(32, 33),
                                        MaskedMLP(32, 32),
                                        MaskedMLP(32, 1),
                                        MaskedMLP(65, 64),
                                        MaskedMLP(64, 16),
                                        MaskedMLP(37, 16),
                                        ])
        self.hard_mask = nn.Sequential(nn.Linear(35,32), activation_func, nn.Linear(32,8),nn.Sigmoid())
        self.hard_mask2 = nn.Sequential(nn.Linear(105,32), activation_func, nn.Linear(32,2),nn.Sigmoid())

        self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                    activation_func,
                                    nn.Linear(32, 33),
                                    activation_func,
                                    )

        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                     activation_func,
                                     nn.Linear(32, 1),
                                     nn.Sigmoid()
                                     )
        self.vis_fc2_sig = nn.Sigmoid()

        self.geometry_fc = nn.Sequential(nn.Linear(32*2+1, 64),
                                         activation_func,
                                         nn.Linear(64, 16),
                                         activation_func)

        self.ray_attention = MultiHeadAttention(4, 16, 4, 4)
        self.out_geometry_fc = nn.Sequential(nn.Linear(16, 16),
                                             activation_func,
                                             nn.Linear(16, 1),
                                             nn.ReLU())

        self.rgb_fc = nn.Sequential(nn.Linear(32+1+4, 16),
                                    activation_func,
                                    nn.Linear(16, 8),
                                    activation_func,
                                    nn.Linear(8, 1))
        # self.rgb_fc = nn.Sequential(nn.Linear(16, 8),
        #                             activation_func,
        #                             nn.Linear(8, 1))

        self.neuray_fc = nn.Sequential(
            nn.Linear(neuray_in_dim, 8,),
            activation_func,
            nn.Linear(8, 1),
        )
        # self.neuray_fc = nn.Sequential(
        #     nn.Linear(8, 1),
        # )

        self.pos_encoding = self.posenc(d_hid=16, n_samples=self.n_samples)

        # self.base_fc.apply(weights_init)
        self.vis_fc2.apply(weights_init)
        self.vis_fc.apply(weights_init)
        # self.geometry_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)
        self.neuray_fc.apply(weights_init)
        self.hyper_linear1.apply(weights_init)
        self.hyper_linear2.apply(weights_init)
        self.hyper_linear3.apply(weights_init)
        self.hard_mask.apply(weights_init)
        self.hard_mask2.apply(weights_init)
        

    def change_pos_encoding(self,n_samples):
        self.pos_encoding = self.posenc(16, n_samples=n_samples)

    def posenc(self, d_hid, n_samples):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).to("cuda:{}".format(0)).float().unsqueeze(0)
        return sinusoid_table

    def forward(self, rgb_feat, neuray_feat, ray_diff, mask, img_feats, que_imgs_info):
        '''
        :param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        :param neuray_feat: [n_rays, n_samples, n_views, 32]
        :param ray_diff: ray direction difference [n_rays, n_samples, n_views, 4], first 3 channels are directions,
        :param img_feats: [3, 32, 136, 180]
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        '''
        #print(rgb_feat.shape, 'rgb_feat', neuray_feat.shape, 'neuray_feat')
        # print(img_feats.shape, que_imgs_info["imgs"].shape, "!!!!!!")
        num_views = rgb_feat.shape[2]
        direction_feat = self.ray_dir_fc(ray_diff)
        rgb_in = rgb_feat[..., :3]
        rgb_feat = rgb_feat + direction_feat
        
        if self.anti_alias_pooling:
            _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
            exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
            weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=2, keepdim=True)[0]) * mask
            weight = weight / (torch.sum(weight, dim=2, keepdim=True) + 1e-8) # means it will trust the one more with more consistent view point
        else:
            weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)

        # neuray layer 0
        # replace neuray_feat
        weight0 = torch.sigmoid(self.neuray_fc(neuray_feat)) * weight # [rn,dn,rfn,f]
        hard_mask = self.hard_mask(direction_feat)
        
        # hard_mask2 = self.hard_mask2(direction_feat.reshape(direction_feat.shape[0],direction_feat.shape[1],-1))
        hyper_matrix = [torch.zeros(1,1)]
        # for i in range(0,1):
        #     mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
        #     mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
        #     mtx = self.hyper_linear3[i](mtx.view(-1))
        #     mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
        #     hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
        # neuray_feat_t = self.act_func(self.masked_mlp[0](neuray_feat, hyper_matrix[0], hard_mask[:,:,:,0]))
        # weight0 = torch.sigmoid(self.neuray_fc(neuray_feat_t)) * weight # [rn,dn,rfn,f]
        # replace neuray_feat
        mean0, var0 = fused_mean_variance(rgb_feat, weight0)  # [n_rays, n_samples, 1, n_feat]
        mean1, var1 = fused_mean_variance(rgb_feat, weight)  # [n_rays, n_samples, 1, n_feat]
        globalfeat = torch.cat([mean0, var0, mean1, var1], dim=-1)  # [n_rays, n_samples, 1, 2*n_feat]

        x = torch.cat([globalfeat.expand(-1, -1, num_views, -1), rgb_feat, neuray_feat], dim=-1)  # [n_rays, n_samples, n_views, 3*n_feat]
        
        # TODO:use hyper linear to replace base_fc    
        # x = self.base_fc(x)
        # print(direction_feat.shape)
        # print(neuray_feat.shape, x.shape,"nr,x")
        for i in range(1,3):
            mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
            mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
            mtx = self.hyper_linear3[i](mtx.view(-1))
            mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
            hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
            # x = self.act_func(torch.matmul(x, hyper_matrix[i]))
            # x = self.act_func(self.masked_mlp[i](x, hyper_matrix[i], torch.ones(hard_mask[:,:,:,i].shape).cuda()))
            x = self.act_func(self.masked_mlp[i](x, hyper_matrix[i], hard_mask[:,:,:,i]))

        # replace vis_fc
        x_vis = self.vis_fc(x * weight)
        # x_t = x * weight
        # for i in range(3,5):
        #     mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
        #     mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
        #     mtx = self.hyper_linear3[i](mtx.view(-1))
        #     mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
        #     hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
        #     # x = self.act_func(torch.matmul(x, hyper_matrix[i]))
        #     x_t = self.act_func(self.masked_mlp[i](x_t, hyper_matrix[i], hard_mask[:,:,:,i]))
        # x_vis = x_t
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1]-1, 1], dim=-1)
        vis = F.sigmoid(vis) * mask
        x = x + x_res
        # replace vis_fc2
        vis = self.vis_fc2(x * vis) * mask
        # x_t = x * vis 
        # # print(x_t.shape, "x_t")
        # for i in range(5,7):
        #     mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
        #     mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
        #     mtx = self.hyper_linear3[i](mtx.view(-1))
        #     mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
        #     hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
        #     # x = self.act_func(torch.matmul(x, hyper_matrix[i]))
        #     if i==4:
        #         x_t = self.act_func(self.masked_mlp[i](x_t, hyper_matrix[i], hard_mask[:,:,:,i]))
        #     else:
        #         x_t = self.vis_fc2_sig(self.masked_mlp[i](x_t, hyper_matrix[i], hard_mask[:,:,:,i]))
        # vis = x_t * mask

        weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)

        mean, var = fused_mean_variance(x, weight) # eliminate the num_view(rfn) dimension here
        globalfeat = torch.cat([mean.squeeze(2), var.squeeze(2), weight.mean(dim=2)], dim=-1)  # [n_rays, n_samples, 32*2+1]
        # replace geometry_fc
        globalfeat = self.geometry_fc(globalfeat)  # [n_rays, n_samples, 16]
        # for i in range(7,9):
        #     mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
        #     mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
        #     mtx = self.hyper_linear3[i](mtx.view(-1))
        #     mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
        #     hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
        #     # x = self.act_func(torch.matmul(x, hyper_matrix[i]))
        #     globalfeat = self.act_func(self.masked_mlp[i](globalfeat, hyper_matrix[i], hard_mask2[:,:,(i-7)]))
        # replace geometry_fc
        num_valid_obs = torch.sum(mask, dim=2)
        globalfeat = globalfeat + self.pos_encoding
        globalfeat, _ = self.ray_attention(globalfeat, globalfeat, globalfeat,
                                           mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
        sigma = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
        sigma_out = sigma.masked_fill(num_valid_obs < 1, 0.)  # set the sigma of invalid point to zero

        # rgb computation
        x = torch.cat([x, vis, ray_diff], dim=-1)
        #replace rgb_fc
        x = self.rgb_fc(x)
        # for i in range(9,10):
        #     mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
        #     mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
        #     mtx = self.hyper_linear3[i](mtx.view(-1))
        #     mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
        #     hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
        #     x = self.act_func(self.masked_mlp[i](x, hyper_matrix[i], hard_mask[:,:,:,(i-2)]))
        # x = self.rgb_fc(x)
        #replace rgb_fc
        x = x.masked_fill(mask == 0, -1e9)
        blending_weights_valid = F.softmax(x, dim=2)  # color blending
        rgb_out = torch.sum(rgb_in*blending_weights_valid, dim=2)
        out = torch.cat([rgb_out, sigma_out], dim=-1)
        return out

class OPONeRFand3MixLinear(nn.Module):
    def __init__(self, neuray_in_dim=32, in_feat_ch=32, n_samples=64, **kwargs):
        super().__init__()
        # self.args = args
        self.anti_alias_pooling = False
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ELU(inplace=True)
        self.n_samples = n_samples
        self.ray_dir_fc = nn.Sequential(nn.Linear(4, 16),
                                        activation_func,
                                        nn.Linear(16, in_feat_ch + 3),
                                        activation_func)

        # self.base_fc = nn.Sequential(nn.Linear((in_feat_ch+3)*5+neuray_in_dim, 64), # 207,64
        #                              activation_func,
        #                              nn.Linear(64, 32),
        #                              activation_func)
        
        #TODO: hyper_linear here
        self.act_func = nn.ELU(inplace=True)
        self.hyper_linear1 = nn.ModuleList([nn.Sequential(nn.Linear(200*200,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU())] * 10)
        # self.hyper_linear1 = nn.ModuleList([nn.Sequential(nn.Linear(136*184,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU())] * 10)
        self.hyper_linear2 = nn.ModuleList([nn.Sequential(nn.Linear(32*128,256), nn.ReLU(), nn.Linear(256,512), nn.ReLU())] * 10)
        self.hyper_linear3 = nn.ModuleList([nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*8), nn.ReLU()), #neuray fc
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 207*64), nn.ReLU()), #base fc
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 64*32), nn.ReLU()),
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*32), nn.ReLU()), #vis fc
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*33), nn.ReLU()),
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*32), nn.ReLU()), #vis fc2
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*1), nn.ReLU()),
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 65*64), nn.ReLU()), #geometry fc
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 64*16), nn.ReLU()), 
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 37*16), nn.ReLU()) #rgb fc
                                            ])
        self.ori_matrix = nn.ParameterList([nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,8]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([207,64]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([64,32]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,32]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,33]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,32]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,1]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([65,64]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([64,16]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([37,16])))
                                            ])

        self.masked_mlp = nn.ModuleList([MaskedMLP(32, 8),
                                        MaskedMLP(207, 64),
                                        MaskedMLP(64, 32),
                                        MaskedMLP(32, 32),
                                        MaskedMLP(32, 33),
                                        MaskedMLP(32, 32),
                                        MaskedMLP(32, 1),
                                        MaskedMLP(65, 64),
                                        MaskedMLP(64, 16),
                                        MaskedMLP(37, 16),
                                        ])
        self.hard_mask = nn.Sequential(nn.Linear(35,32), activation_func, nn.Linear(32,8),nn.Sigmoid())
        self.hard_mask2 = nn.Sequential(nn.Linear(105,32), activation_func, nn.Linear(32,2),nn.Sigmoid())

        self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                    activation_func,
                                    nn.Linear(32, 33),
                                    activation_func,
                                    )

        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                     activation_func,
                                     nn.Linear(32, 1),
                                     nn.Sigmoid()
                                     )
        self.vis_fc2_sig = nn.Sigmoid()

        self.geometry_fc = nn.Sequential(nn.Linear(32*2+1, 64),
                                         activation_func,
                                         nn.Linear(64, 16),
                                         activation_func)

        self.ray_attention = MultiHeadAttention(4, 16, 4, 4)
        self.out_geometry_fc = nn.Sequential(nn.Linear(16, 16),
                                             activation_func,
                                             nn.Linear(16, 1),
                                             nn.ReLU())

        self.rgb_fc = nn.Sequential(nn.Linear(32+1+4, 16),
                                    activation_func,
                                    nn.Linear(16, 8),
                                    activation_func,
                                    nn.Linear(8, 1))
        # self.rgb_fc = nn.Sequential(nn.Linear(16, 8),
        #                             activation_func,
        #                             nn.Linear(8, 1))

        self.neuray_fc = nn.Sequential(
            nn.Linear(neuray_in_dim, 8,),
            activation_func,
            nn.Linear(8, 1),
        )
        # self.neuray_fc = nn.Sequential(
        #     nn.Linear(8, 1),
        # )

        self.pos_encoding = self.posenc(d_hid=16, n_samples=self.n_samples)

        # self.base_fc.apply(weights_init)
        self.vis_fc2.apply(weights_init)
        self.vis_fc.apply(weights_init)
        # self.geometry_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)
        self.neuray_fc.apply(weights_init)
        self.hyper_linear1.apply(weights_init)
        self.hyper_linear2.apply(weights_init)
        self.hyper_linear3.apply(weights_init)
        self.hard_mask.apply(weights_init)
        self.hard_mask2.apply(weights_init)
        

    def change_pos_encoding(self,n_samples):
        self.pos_encoding = self.posenc(16, n_samples=n_samples)

    def posenc(self, d_hid, n_samples):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).to("cuda:{}".format(0)).float().unsqueeze(0)
        return sinusoid_table

    def forward(self, rgb_feat, neuray_feat, ray_diff, mask, img_feats, que_imgs_info):
        '''
        :param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        :param neuray_feat: [n_rays, n_samples, n_views, 32]
        :param ray_diff: ray direction difference [n_rays, n_samples, n_views, 4], first 3 channels are directions,
        :param img_feats: [3, 32, 136, 180]
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        '''
        #print(rgb_feat.shape, 'rgb_feat', neuray_feat.shape, 'neuray_feat')
        # print(img_feats.shape, que_imgs_info["imgs"].shape, "!!!!!!")
        num_views = rgb_feat.shape[2]
        direction_feat = self.ray_dir_fc(ray_diff)
        rgb_in = rgb_feat[..., :3]
        rgb_feat = rgb_feat + direction_feat
        if self.anti_alias_pooling:
            _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
            exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
            weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=2, keepdim=True)[0]) * mask
            weight = weight / (torch.sum(weight, dim=2, keepdim=True) + 1e-8) # means it will trust the one more with more consistent view point
        else:
            weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)

        # neuray layer 0
        # replace neuray_feat
        weight0 = torch.sigmoid(self.neuray_fc(neuray_feat)) * weight # [rn,dn,rfn,f]
        hard_mask = self.hard_mask(direction_feat)
        # hard_mask2 = self.hard_mask2(direction_feat.reshape(direction_feat.shape[0],direction_feat.shape[1],-1))
        hyper_matrix = [torch.zeros(1,1)]
        # for i in range(0,1):
        #     mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
        #     mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
        #     mtx = self.hyper_linear3[i](mtx.view(-1))
        #     mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
        #     hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
        # neuray_feat_t = self.act_func(self.masked_mlp[0](neuray_feat, hyper_matrix[0], hard_mask[:,:,:,0]))
        # weight0 = torch.sigmoid(self.neuray_fc(neuray_feat_t)) * weight # [rn,dn,rfn,f]
        # replace neuray_feat
        mean0, var0 = fused_mean_variance(rgb_feat, weight0)  # [n_rays, n_samples, 1, n_feat]
        mean1, var1 = fused_mean_variance(rgb_feat, weight)  # [n_rays, n_samples, 1, n_feat]
        globalfeat = torch.cat([mean0, var0, mean1, var1], dim=-1)  # [n_rays, n_samples, 1, 2*n_feat]

        x = torch.cat([globalfeat.expand(-1, -1, num_views, -1), rgb_feat, neuray_feat], dim=-1)  # [n_rays, n_samples, n_views, 3*n_feat]
        
        # TODO:use hyper linear to replace base_fc    
        # x = self.base_fc(x)
        # print(direction_feat.shape)
        # print(neuray_feat.shape, x.shape,"nr,x")
        for i in range(1,3):
            mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
            mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
            mtx = self.hyper_linear3[i](mtx.view(-1))
            mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
            hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
            # x = self.act_func(torch.matmul(x, hyper_matrix[i]))
            x = self.act_func(self.masked_mlp[i](x, hyper_matrix[i], torch.ones(hard_mask[:,:,:,i].shape).cuda()))
            # x = self.act_func(self.masked_mlp[i](x, hyper_matrix[i], hard_mask[:,:,:,i]))

        # replace vis_fc
        x_vis = self.vis_fc(x * weight)
        # x_t = x * weight
        # for i in range(3,5):
        #     mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
        #     mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
        #     mtx = self.hyper_linear3[i](mtx.view(-1))
        #     mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
        #     hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
        #     # x = self.act_func(torch.matmul(x, hyper_matrix[i]))
        #     x_t = self.act_func(self.masked_mlp[i](x_t, hyper_matrix[i], hard_mask[:,:,:,i]))
        # x_vis = x_t
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1]-1, 1], dim=-1)
        vis = F.sigmoid(vis) * mask
        x = x + x_res
        # replace vis_fc2
        vis = self.vis_fc2(x * vis) * mask
        # x_t = x * vis 
        # # print(x_t.shape, "x_t")
        # for i in range(5,7):
        #     mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
        #     mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
        #     mtx = self.hyper_linear3[i](mtx.view(-1))
        #     mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
        #     hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
        #     # x = self.act_func(torch.matmul(x, hyper_matrix[i]))
        #     if i==4:
        #         x_t = self.act_func(self.masked_mlp[i](x_t, hyper_matrix[i], hard_mask[:,:,:,i]))
        #     else:
        #         x_t = self.vis_fc2_sig(self.masked_mlp[i](x_t, hyper_matrix[i], hard_mask[:,:,:,i]))
        # vis = x_t * mask

        weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)

        mean, var = fused_mean_variance(x, weight) # eliminate the num_view(rfn) dimension here
        globalfeat = torch.cat([mean.squeeze(2), var.squeeze(2), weight.mean(dim=2)], dim=-1)  # [n_rays, n_samples, 32*2+1]
        # replace geometry_fc
        globalfeat = self.geometry_fc(globalfeat)  # [n_rays, n_samples, 16]
        # for i in range(7,9):
        #     mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
        #     mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
        #     mtx = self.hyper_linear3[i](mtx.view(-1))
        #     mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
        #     hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
        #     # x = self.act_func(torch.matmul(x, hyper_matrix[i]))
        #     globalfeat = self.act_func(self.masked_mlp[i](globalfeat, hyper_matrix[i], hard_mask2[:,:,(i-7)]))
        # replace geometry_fc
        num_valid_obs = torch.sum(mask, dim=2)
        globalfeat = globalfeat + self.pos_encoding
        globalfeat, _ = self.ray_attention(globalfeat, globalfeat, globalfeat,
                                           mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
        sigma = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
        sigma_out = sigma.masked_fill(num_valid_obs < 1, 0.)  # set the sigma of invalid point to zero

        # rgb computation
        x = torch.cat([x, vis, ray_diff], dim=-1)
        #replace rgb_fc
        x = self.rgb_fc(x)
        # for i in range(9,10):
        #     mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
        #     mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
        #     mtx = self.hyper_linear3[i](mtx.view(-1))
        #     mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
        #     hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
        #     x = self.act_func(self.masked_mlp[i](x, hyper_matrix[i], hard_mask[:,:,:,(i-2)]))
        # x = self.rgb_fc(x)
        #replace rgb_fc
        x = x.masked_fill(mask == 0, -1e9)
        blending_weights_valid = F.softmax(x, dim=2)  # color blending
        rgb_out = torch.sum(rgb_in*blending_weights_valid, dim=2)
        out = torch.cat([rgb_out, sigma_out], dim=-1)
        return out

class MaskedMLP2(nn.Module):
    def __init__(self, in_size, out_size):
        super(MaskedMLP2, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.Tensor(out_size, in_size))
        self.bias = nn.Parameter(torch.Tensor(out_size))
        self.threshold = nn.Parameter(torch.Tensor(out_size, in_size))
        self.step = BinaryStep.apply
        #self.mask = None
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound) 
        with torch.no_grad():
            #std = self.weight.std()
            # nn.init.kaiming_uniform_(self.threshold, a=math.sqrt(5))
            self.threshold.data.fill_(0.)
    
    def forward(self, input, weight_matrix, hard_mask):
        # abs_weight = torch.abs(self.weight)
        # # print(abs_weight)
        # # print(self.threshold)
        # threshold = self.threshold.view(abs_weight.shape[0], -1)
        # abs_weight = abs_weight-threshold
        # mask = self.step(abs_weight)
        # ratio = torch.sum(mask) / mask.numel()
        # #print("keep ratio {:.2f}".format(ratio))
        # if ratio <= 0.01:
        #     with torch.no_grad():
        #         #std = self.weight.std()
        #         # self.threshold.data.fill_(0.)
        #         nn.init.kaiming_uniform_(self.threshold, a=math.sqrt(5))
        #     abs_weight = torch.abs(self.weight)
        #     threshold = self.threshold.view(abs_weight.shape[0], -1)
        #     abs_weight = abs_weight-threshold
        #     mask = self.step(abs_weight)
        # # print(mask.shape,mask)
        # # 0 noplus 1 plus
        # # print(self.weight.shape,mask.shape,weight_matrix.shape)
        # masked_weight = self.weight*mask + weight_matrix.transpose(0,1) * (1-mask)
        masked_weight = weight_matrix.transpose(0,1)
        # print(input.shape,masked_weight.shape)
        output1 = torch.nn.functional.linear(input, masked_weight, self.bias)
        output2 = torch.nn.functional.linear(input, self.weight, self.bias)
        # print(hard_mask.shape, output1.shape, output2.shape)
        return output1*hard_mask.unsqueeze(dim=-1) + output2*(1-hard_mask).unsqueeze(dim=-1)

class OPONeRFand5MixLinear(nn.Module):
    def __init__(self, neuray_in_dim=32, in_feat_ch=32, n_samples=64, **kwargs):
        super().__init__()
        # self.args = args
        self.anti_alias_pooling = False
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ELU(inplace=True)
        self.n_samples = n_samples
        self.ray_dir_fc = nn.Sequential(nn.Linear(4, 16),
                                        activation_func,
                                        nn.Linear(16, in_feat_ch + 3),
                                        activation_func)

        # self.base_fc = nn.Sequential(nn.Linear((in_feat_ch+3)*5+neuray_in_dim, 64), # 207,64
        #                              activation_func,
        #                              nn.Linear(64, 32),
        #                              activation_func)
        
        #TODO: hyper_linear here
        self.act_func = nn.ELU(inplace=True)
        self.hyper_linear1 = nn.ModuleList([nn.Sequential(nn.Linear(200*200,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU())] * 10)
        # self.hyper_linear1 = nn.ModuleList([nn.Sequential(nn.Linear(136*184,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU())] * 10)
        self.hyper_linear2 = nn.ModuleList([nn.Sequential(nn.Linear(32*128,256), nn.ReLU(), nn.Linear(256,512), nn.ReLU())] * 10)
        self.hyper_linear3 = nn.ModuleList([nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*8), nn.ReLU()), #neuray fc
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 207*64), nn.ReLU()), #base fc
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 64*32), nn.ReLU()),
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*32), nn.ReLU()), #vis fc
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*33), nn.ReLU()),
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*32), nn.ReLU()), #vis fc2
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*1), nn.ReLU()),
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 65*64), nn.ReLU()), #geometry fc
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 64*16), nn.ReLU()), 
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 37*16), nn.ReLU()) #rgb fc
                                            ])
        self.ori_matrix = nn.ParameterList([nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,8]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([207,64]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([64,32]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,32]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,33]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,32]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,1]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([65,64]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([64,16]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([37,16])))
                                            ])

        self.masked_mlp = nn.ModuleList([MaskedMLP2(32, 8),
                                        MaskedMLP2(207, 64),
                                        MaskedMLP2(64, 32),
                                        MaskedMLP2(32, 32),
                                        MaskedMLP2(32, 33),
                                        MaskedMLP2(32, 32),
                                        MaskedMLP2(32, 1),
                                        MaskedMLP2(65, 64),
                                        MaskedMLP2(64, 16),
                                        MaskedMLP2(37, 16),
                                        ])
        self.hard_mask = nn.Sequential(nn.Linear(35,32), activation_func, nn.Linear(32,8),nn.Sigmoid())
        self.hard_mask2 = nn.Sequential(nn.Linear(105,32), activation_func, nn.Linear(32,2),nn.Sigmoid())

        self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                    activation_func,
                                    nn.Linear(32, 33),
                                    activation_func,
                                    )

        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                     activation_func,
                                     nn.Linear(32, 1),
                                     nn.Sigmoid()
                                     )
        self.vis_fc2_sig = nn.Sigmoid()

        self.geometry_fc = nn.Sequential(nn.Linear(32*2+1, 64),
                                         activation_func,
                                         nn.Linear(64, 16),
                                         activation_func)

        self.ray_attention = MultiHeadAttention(4, 16, 4, 4)
        self.out_geometry_fc = nn.Sequential(nn.Linear(16, 16),
                                             activation_func,
                                             nn.Linear(16, 1),
                                             nn.ReLU())

        self.rgb_fc = nn.Sequential(nn.Linear(32+1+4, 16),
                                    activation_func,
                                    nn.Linear(16, 8),
                                    activation_func,
                                    nn.Linear(8, 1))
        # self.rgb_fc = nn.Sequential(nn.Linear(16, 8),
        #                             activation_func,
        #                             nn.Linear(8, 1))

        self.neuray_fc = nn.Sequential(
            nn.Linear(neuray_in_dim, 8,),
            activation_func,
            nn.Linear(8, 1),
        )
        # self.neuray_fc = nn.Sequential(
        #     nn.Linear(8, 1),
        # )

        self.pos_encoding = self.posenc(d_hid=16, n_samples=self.n_samples)

        # self.base_fc.apply(weights_init)
        self.vis_fc2.apply(weights_init)
        self.vis_fc.apply(weights_init)
        # self.geometry_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)
        self.neuray_fc.apply(weights_init)
        self.hyper_linear1.apply(weights_init)
        self.hyper_linear2.apply(weights_init)
        self.hyper_linear3.apply(weights_init)
        self.hard_mask.apply(weights_init)
        self.hard_mask2.apply(weights_init)
        

    def change_pos_encoding(self,n_samples):
        self.pos_encoding = self.posenc(16, n_samples=n_samples)

    def posenc(self, d_hid, n_samples):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).to("cuda:{}".format(0)).float().unsqueeze(0)
        return sinusoid_table

    def forward(self, rgb_feat, neuray_feat, ray_diff, mask, img_feats, que_imgs_info):
        '''
        :param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        :param neuray_feat: [n_rays, n_samples, n_views, 32]
        :param ray_diff: ray direction difference [n_rays, n_samples, n_views, 4], first 3 channels are directions,
        :param img_feats: [3, 32, 136, 180]
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        '''
        #print(rgb_feat.shape, 'rgb_feat', neuray_feat.shape, 'neuray_feat')
        # print(img_feats.shape, que_imgs_info["imgs"].shape, "!!!!!!")
        num_views = rgb_feat.shape[2]
        direction_feat = self.ray_dir_fc(ray_diff)
        rgb_in = rgb_feat[..., :3]
        rgb_feat = rgb_feat + direction_feat
        if self.anti_alias_pooling:
            _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
            exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
            weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=2, keepdim=True)[0]) * mask
            weight = weight / (torch.sum(weight, dim=2, keepdim=True) + 1e-8) # means it will trust the one more with more consistent view point
        else:
            weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)

        # neuray layer 0
        # replace neuray_feat
        weight0 = torch.sigmoid(self.neuray_fc(neuray_feat)) * weight # [rn,dn,rfn,f]
        hard_mask = self.hard_mask(direction_feat)
        # hard_mask2 = self.hard_mask2(direction_feat.reshape(direction_feat.shape[0],direction_feat.shape[1],-1))
        hyper_matrix = [torch.zeros(1,1)]
        # for i in range(0,1):
        #     mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
        #     mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
        #     mtx = self.hyper_linear3[i](mtx.view(-1))
        #     mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
        #     hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
        # neuray_feat_t = self.act_func(self.masked_mlp[0](neuray_feat, hyper_matrix[0], hard_mask[:,:,:,0]))
        # weight0 = torch.sigmoid(self.neuray_fc(neuray_feat_t)) * weight # [rn,dn,rfn,f]
        # replace neuray_feat
        mean0, var0 = fused_mean_variance(rgb_feat, weight0)  # [n_rays, n_samples, 1, n_feat]
        mean1, var1 = fused_mean_variance(rgb_feat, weight)  # [n_rays, n_samples, 1, n_feat]
        globalfeat = torch.cat([mean0, var0, mean1, var1], dim=-1)  # [n_rays, n_samples, 1, 2*n_feat]

        x = torch.cat([globalfeat.expand(-1, -1, num_views, -1), rgb_feat, neuray_feat], dim=-1)  # [n_rays, n_samples, n_views, 3*n_feat]
        
        # TODO:use hyper linear to replace base_fc    
        # x = self.base_fc(x)
        # print(direction_feat.shape)
        # print(neuray_feat.shape, x.shape,"nr,x")
        for i in range(1,3):
            mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
            mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
            mtx = self.hyper_linear3[i](mtx.view(-1))
            mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
            hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
            # x = self.act_func(torch.matmul(x, hyper_matrix[i]))
            x = self.act_func(self.masked_mlp[i](x, hyper_matrix[i], torch.ones(hard_mask[:,:,:,i].shape).cuda()))
            # x = self.act_func(self.masked_mlp[i](x, hyper_matrix[i], hard_mask[:,:,:,i]))

        # replace vis_fc
        x_vis = self.vis_fc(x * weight)
        # x_t = x * weight
        # for i in range(3,5):
        #     mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
        #     mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
        #     mtx = self.hyper_linear3[i](mtx.view(-1))
        #     mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
        #     hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
        #     # x = self.act_func(torch.matmul(x, hyper_matrix[i]))
        #     x_t = self.act_func(self.masked_mlp[i](x_t, hyper_matrix[i], hard_mask[:,:,:,i]))
        # x_vis = x_t
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1]-1, 1], dim=-1)
        vis = F.sigmoid(vis) * mask
        x = x + x_res
        # replace vis_fc2
        vis = self.vis_fc2(x * vis) * mask
        # x_t = x * vis 
        # # print(x_t.shape, "x_t")
        # for i in range(5,7):
        #     mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
        #     mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
        #     mtx = self.hyper_linear3[i](mtx.view(-1))
        #     mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
        #     hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
        #     # x = self.act_func(torch.matmul(x, hyper_matrix[i]))
        #     if i==4:
        #         x_t = self.act_func(self.masked_mlp[i](x_t, hyper_matrix[i], hard_mask[:,:,:,i]))
        #     else:
        #         x_t = self.vis_fc2_sig(self.masked_mlp[i](x_t, hyper_matrix[i], hard_mask[:,:,:,i]))
        # vis = x_t * mask

        weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)

        mean, var = fused_mean_variance(x, weight) # eliminate the num_view(rfn) dimension here
        globalfeat = torch.cat([mean.squeeze(2), var.squeeze(2), weight.mean(dim=2)], dim=-1)  # [n_rays, n_samples, 32*2+1]
        # replace geometry_fc
        globalfeat = self.geometry_fc(globalfeat)  # [n_rays, n_samples, 16]
        # for i in range(7,9):
        #     mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
        #     mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
        #     mtx = self.hyper_linear3[i](mtx.view(-1))
        #     mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
        #     hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
        #     # x = self.act_func(torch.matmul(x, hyper_matrix[i]))
        #     globalfeat = self.act_func(self.masked_mlp[i](globalfeat, hyper_matrix[i], hard_mask2[:,:,(i-7)]))
        # replace geometry_fc
        num_valid_obs = torch.sum(mask, dim=2)
        globalfeat = globalfeat + self.pos_encoding
        globalfeat, _ = self.ray_attention(globalfeat, globalfeat, globalfeat,
                                           mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
        sigma = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
        sigma_out = sigma.masked_fill(num_valid_obs < 1, 0.)  # set the sigma of invalid point to zero

        # rgb computation
        x = torch.cat([x, vis, ray_diff], dim=-1)
        #replace rgb_fc
        x = self.rgb_fc(x)
        # for i in range(9,10):
        #     mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
        #     mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
        #     mtx = self.hyper_linear3[i](mtx.view(-1))
        #     mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
        #     hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
        #     x = self.act_func(self.masked_mlp[i](x, hyper_matrix[i], hard_mask[:,:,:,(i-2)]))
        # x = self.rgb_fc(x)
        #replace rgb_fc
        x = x.masked_fill(mask == 0, -1e9)
        blending_weights_valid = F.softmax(x, dim=2)  # color blending
        rgb_out = torch.sum(rgb_in*blending_weights_valid, dim=2)
        out = torch.cat([rgb_out, sigma_out], dim=-1)
        return out

class MaskedMLP3(nn.Module):
    def __init__(self, in_size, out_size):
        super(MaskedMLP3, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.Tensor(out_size, in_size))
        self.bias = nn.Parameter(torch.Tensor(out_size))
        self.threshold = nn.Parameter(torch.Tensor(out_size, in_size))
        self.step = BinaryStep.apply
        #self.mask = None
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound) 
        with torch.no_grad():
            #std = self.weight.std()
            # nn.init.kaiming_uniform_(self.threshold, a=math.sqrt(5))
            self.threshold.data.fill_(0.)
    
    def forward(self, input, weight_matrix, hard_mask):
        abs_weight = torch.abs(self.weight)
        # # print(abs_weight)
        # # print(self.threshold)
        threshold = self.threshold.view(abs_weight.shape[0], -1)
        # abs_weight = abs_weight-threshold
        # mask = self.step(abs_weight)
        # ratio = torch.sum(mask) / mask.numel()
        # #print("keep ratio {:.2f}".format(ratio))
        # if ratio <= 0.01:
        #     with torch.no_grad():
        #         #std = self.weight.std()
        #         # self.threshold.data.fill_(0.)
        #         nn.init.kaiming_uniform_(self.threshold, a=math.sqrt(5))
        #     abs_weight = torch.abs(self.weight)
        #     threshold = self.threshold.view(abs_weight.shape[0], -1)
        #     abs_weight = abs_weight-threshold
        #     mask = self.step(abs_weight)
        # # print(mask.shape,mask)
        # # 0 noplus 1 plus
        # # print(self.weight.shape,mask.shape,weight_matrix.shape)
        masked_weight = self.weight*threshold + weight_matrix.transpose(0,1) * (1-threshold)
        # masked_weight = self.weight*mask + weight_matrix.transpose(0,1) * (1-mask)
        # print(input.shape,masked_weight.shape)
        output1 = torch.nn.functional.linear(input, masked_weight, self.bias)
        output2 = torch.nn.functional.linear(input, self.weight, self.bias)
        # print(hard_mask.shape, output1.shape, output2.shape)
        return output1*hard_mask.unsqueeze(dim=-1) + output2*(1-hard_mask).unsqueeze(dim=-1)

class OPONeRFand7MixLinear(nn.Module):
    def __init__(self, neuray_in_dim=32, in_feat_ch=32, n_samples=64, **kwargs):
        super().__init__()
        # self.args = args
        self.anti_alias_pooling = False
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ELU(inplace=True)
        self.n_samples = n_samples
        self.ray_dir_fc = nn.Sequential(nn.Linear(4, 16),
                                        activation_func,
                                        nn.Linear(16, in_feat_ch + 3),
                                        activation_func)

        # self.base_fc = nn.Sequential(nn.Linear((in_feat_ch+3)*5+neuray_in_dim, 64), # 207,64
        #                              activation_func,
        #                              nn.Linear(64, 32),
        #                              activation_func)
        
        #TODO: hyper_linear here
        self.act_func = nn.ELU(inplace=True)
        self.hyper_linear1 = nn.ModuleList([nn.Sequential(nn.Linear(200*200,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU())] * 10)
        # self.hyper_linear1 = nn.ModuleList([nn.Sequential(nn.Linear(136*184,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU())] * 10)
        self.hyper_linear2 = nn.ModuleList([nn.Sequential(nn.Linear(32*128,256), nn.ReLU(), nn.Linear(256,512), nn.ReLU())] * 10)
        self.hyper_linear3 = nn.ModuleList([nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*8), nn.ReLU()), #neuray fc
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 207*64), nn.ReLU()), #base fc
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 64*32), nn.ReLU()),
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*32), nn.ReLU()), #vis fc
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*33), nn.ReLU()),
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*32), nn.ReLU()), #vis fc2
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 32*1), nn.ReLU()),
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 65*64), nn.ReLU()), #geometry fc
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 64*16), nn.ReLU()), 
                                            nn.Sequential(nn.Linear(3*512,256), nn.ReLU(), nn.Linear(256, 37*16), nn.ReLU()) #rgb fc
                                            ])
        self.ori_matrix = nn.ParameterList([nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,8]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([207,64]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([64,32]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,32]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,33]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,32]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([32,1]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([65,64]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([64,16]))),
                                            nn.Parameter(nn.init.kaiming_normal_(torch.zeros([37,16])))
                                            ])

        self.masked_mlp = nn.ModuleList([MaskedMLP2(32, 8),
                                        MaskedMLP2(207, 64),
                                        MaskedMLP2(64, 32),
                                        MaskedMLP2(32, 32),
                                        MaskedMLP2(32, 33),
                                        MaskedMLP2(32, 32),
                                        MaskedMLP2(32, 1),
                                        MaskedMLP2(65, 64),
                                        MaskedMLP2(64, 16),
                                        MaskedMLP2(37, 16),
                                        ])
        self.hard_mask = nn.Sequential(nn.Linear(35,32), activation_func, nn.Linear(32,8),nn.Sigmoid())
        self.hard_mask2 = nn.Sequential(nn.Linear(105,32), activation_func, nn.Linear(32,2),nn.Sigmoid())

        self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                    activation_func,
                                    nn.Linear(32, 33),
                                    activation_func,
                                    )

        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                     activation_func,
                                     nn.Linear(32, 1),
                                     nn.Sigmoid()
                                     )
        self.vis_fc2_sig = nn.Sigmoid()

        self.geometry_fc = nn.Sequential(nn.Linear(32*2+1, 64),
                                         activation_func,
                                         nn.Linear(64, 16),
                                         activation_func)

        self.ray_attention = MultiHeadAttention(4, 16, 4, 4)
        self.out_geometry_fc = nn.Sequential(nn.Linear(16, 16),
                                             activation_func,
                                             nn.Linear(16, 1),
                                             nn.ReLU())

        self.rgb_fc = nn.Sequential(nn.Linear(32+1+4, 16),
                                    activation_func,
                                    nn.Linear(16, 8),
                                    activation_func,
                                    nn.Linear(8, 1))
        # self.rgb_fc = nn.Sequential(nn.Linear(16, 8),
        #                             activation_func,
        #                             nn.Linear(8, 1))

        self.neuray_fc = nn.Sequential(
            nn.Linear(neuray_in_dim, 8,),
            activation_func,
            nn.Linear(8, 1),
        )
        # self.neuray_fc = nn.Sequential(
        #     nn.Linear(8, 1),
        # )

        self.pos_encoding = self.posenc(d_hid=16, n_samples=self.n_samples)

        # self.base_fc.apply(weights_init)
        self.vis_fc2.apply(weights_init)
        self.vis_fc.apply(weights_init)
        # self.geometry_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)
        self.neuray_fc.apply(weights_init)
        self.hyper_linear1.apply(weights_init)
        self.hyper_linear2.apply(weights_init)
        self.hyper_linear3.apply(weights_init)
        self.hard_mask.apply(weights_init)
        self.hard_mask2.apply(weights_init)
        

    def change_pos_encoding(self,n_samples):
        self.pos_encoding = self.posenc(16, n_samples=n_samples)

    def posenc(self, d_hid, n_samples):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).to("cuda:{}".format(0)).float().unsqueeze(0)
        return sinusoid_table

    def forward(self, rgb_feat, neuray_feat, ray_diff, mask, img_feats, que_imgs_info):
        '''
        :param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        :param neuray_feat: [n_rays, n_samples, n_views, 32]
        :param ray_diff: ray direction difference [n_rays, n_samples, n_views, 4], first 3 channels are directions,
        :param img_feats: [3, 32, 136, 180]
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        '''
        #print(rgb_feat.shape, 'rgb_feat', neuray_feat.shape, 'neuray_feat')
        # print(img_feats.shape, que_imgs_info["imgs"].shape, "!!!!!!")
        num_views = rgb_feat.shape[2]
        direction_feat = self.ray_dir_fc(ray_diff)
        rgb_in = rgb_feat[..., :3]
        rgb_feat = rgb_feat + direction_feat
        if self.anti_alias_pooling:
            _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
            exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
            weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=2, keepdim=True)[0]) * mask
            weight = weight / (torch.sum(weight, dim=2, keepdim=True) + 1e-8) # means it will trust the one more with more consistent view point
        else:
            weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)

        # neuray layer 0
        # replace neuray_feat
        weight0 = torch.sigmoid(self.neuray_fc(neuray_feat)) * weight # [rn,dn,rfn,f]
        hard_mask = self.hard_mask(direction_feat)
        # hard_mask2 = self.hard_mask2(direction_feat.reshape(direction_feat.shape[0],direction_feat.shape[1],-1))
        hyper_matrix = [torch.zeros(1,1)]
        # for i in range(0,1):
        #     mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
        #     mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
        #     mtx = self.hyper_linear3[i](mtx.view(-1))
        #     mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
        #     hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
        # neuray_feat_t = self.act_func(self.masked_mlp[0](neuray_feat, hyper_matrix[0], hard_mask[:,:,:,0]))
        # weight0 = torch.sigmoid(self.neuray_fc(neuray_feat_t)) * weight # [rn,dn,rfn,f]
        # replace neuray_feat
        mean0, var0 = fused_mean_variance(rgb_feat, weight0)  # [n_rays, n_samples, 1, n_feat]
        mean1, var1 = fused_mean_variance(rgb_feat, weight)  # [n_rays, n_samples, 1, n_feat]
        globalfeat = torch.cat([mean0, var0, mean1, var1], dim=-1)  # [n_rays, n_samples, 1, 2*n_feat]

        x = torch.cat([globalfeat.expand(-1, -1, num_views, -1), rgb_feat, neuray_feat], dim=-1)  # [n_rays, n_samples, n_views, 3*n_feat]
        
        # TODO:use hyper linear to replace base_fc    
        # x = self.base_fc(x)
        # print(direction_feat.shape)
        # print(neuray_feat.shape, x.shape,"nr,x")
        for i in range(1,3):
            mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
            mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
            mtx = self.hyper_linear3[i](mtx.view(-1))
            mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
            hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
            # x = self.act_func(torch.matmul(x, hyper_matrix[i]))
            # x = self.act_func(self.masked_mlp[i](x, hyper_matrix[i], torch.ones(hard_mask[:,:,:,i].shape).cuda()))
            x = self.act_func(self.masked_mlp[i](x, hyper_matrix[i], hard_mask[:,:,:,i]))

        # replace vis_fc
        x_vis = self.vis_fc(x * weight)
        # x_t = x * weight
        # for i in range(3,5):
        #     mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
        #     mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
        #     mtx = self.hyper_linear3[i](mtx.view(-1))
        #     mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
        #     hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
        #     # x = self.act_func(torch.matmul(x, hyper_matrix[i]))
        #     x_t = self.act_func(self.masked_mlp[i](x_t, hyper_matrix[i], hard_mask[:,:,:,i]))
        # x_vis = x_t
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1]-1, 1], dim=-1)
        vis = F.sigmoid(vis) * mask
        x = x + x_res
        # replace vis_fc2
        vis = self.vis_fc2(x * vis) * mask
        # x_t = x * vis 
        # # print(x_t.shape, "x_t")
        # for i in range(5,7):
        #     mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
        #     mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
        #     mtx = self.hyper_linear3[i](mtx.view(-1))
        #     mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
        #     hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
        #     # x = self.act_func(torch.matmul(x, hyper_matrix[i]))
        #     if i==4:
        #         x_t = self.act_func(self.masked_mlp[i](x_t, hyper_matrix[i], hard_mask[:,:,:,i]))
        #     else:
        #         x_t = self.vis_fc2_sig(self.masked_mlp[i](x_t, hyper_matrix[i], hard_mask[:,:,:,i]))
        # vis = x_t * mask

        weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)

        mean, var = fused_mean_variance(x, weight) # eliminate the num_view(rfn) dimension here
        globalfeat = torch.cat([mean.squeeze(2), var.squeeze(2), weight.mean(dim=2)], dim=-1)  # [n_rays, n_samples, 32*2+1]
        # replace geometry_fc
        globalfeat = self.geometry_fc(globalfeat)  # [n_rays, n_samples, 16]
        # for i in range(7,9):
        #     mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
        #     mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
        #     mtx = self.hyper_linear3[i](mtx.view(-1))
        #     mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
        #     hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
        #     # x = self.act_func(torch.matmul(x, hyper_matrix[i]))
        #     globalfeat = self.act_func(self.masked_mlp[i](globalfeat, hyper_matrix[i], hard_mask2[:,:,(i-7)]))
        # replace geometry_fc
        num_valid_obs = torch.sum(mask, dim=2)
        globalfeat = globalfeat + self.pos_encoding
        globalfeat, _ = self.ray_attention(globalfeat, globalfeat, globalfeat,
                                           mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
        sigma = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
        sigma_out = sigma.masked_fill(num_valid_obs < 1, 0.)  # set the sigma of invalid point to zero

        # rgb computation
        x = torch.cat([x, vis, ray_diff], dim=-1)
        #replace rgb_fc
        x = self.rgb_fc(x)
        # for i in range(9,10):
        #     mtx = self.hyper_linear1[i](img_feats.view(img_feats.shape[0],img_feats.shape[1],-1))
        #     mtx = self.hyper_linear2[i](mtx.view(mtx.shape[0],-1))
        #     mtx = self.hyper_linear3[i](mtx.view(-1))
        #     mtx = mtx.reshape(self.ori_matrix[i].shape[0],self.ori_matrix[i].shape[1])
        #     hyper_matrix.append(torch.mul(self.ori_matrix[i], mtx))
        #     x = self.act_func(self.masked_mlp[i](x, hyper_matrix[i], hard_mask[:,:,:,(i-2)]))
        # x = self.rgb_fc(x)
        #replace rgb_fc
        x = x.masked_fill(mask == 0, -1e9)
        blending_weights_valid = F.softmax(x, dim=2)  # color blending
        rgb_out = torch.sum(rgb_in*blending_weights_valid, dim=2)
        out = torch.cat([rgb_out, sigma_out], dim=-1)
        return out