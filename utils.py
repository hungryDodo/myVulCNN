import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
from FcaNet import MultiSpectralAttentionLayer

class Res18_Test(nn.Module):
    def __init__(self, num_classes):
        super(Res18_Test, self).__init__()
        self.res18_encoder = resnet18(num_classes=num_classes)
        
    def forward(self, x):
        return self.res18_encoder(x.float()), 0


class TextCNNx2(nn.Module):
    def __init__(self, hidden_size):
        super(TextCNNx2, self).__init__()
        self.filter_sizes = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.num_filters = 32
        classifier_dropout = 0.1
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(3, self.num_filters, (k, hidden_size // 4), (1, 2)) for k in self.filter_sizes])
        self.convs2 = nn.ModuleList(
            [nn.Conv2d(self.num_filters, self.num_filters * 2, (k, hidden_size * 3 // 8 + 1)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(classifier_dropout)
        num_classes = 2
        self.fc = nn.Linear(self.num_filters * 2 * len(self.filter_sizes), num_classes)

    def conv_and_pool(self, x, conv1, conv2):
        x = conv1(x)
        x = conv2(x)
        x = F.relu(x).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = x.float()
        # out = out.unsqueeze(1)
        hidden_state = torch.cat([
            self.conv_and_pool(out, self.convs1[i], self.convs2[i]) for i in range(len(self.filter_sizes))
            ], 1)
        out = self.dropout(hidden_state)
        out = self.fc(out)
        return out, hidden_state


class MHA_Block(nn.Module):
    def __init__(self, hidden_size):
        super(MHA_Block, self).__init__()
        self.num_heads = 8
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.post_attn_fc = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        batch_size, channel_num, width, dim = x.size()
        dim_per_head = dim // self.num_heads
        q = q.reshape(batch_size, channel_num, width, self.num_heads, dim_per_head).transpose(1, 3).reshape(-1, width, dim_per_head)
        k = k.reshape(batch_size, channel_num, width, self.num_heads, dim_per_head).transpose(1, 3).reshape(-1, width, dim_per_head)
        v = v.reshape(batch_size, channel_num, width, self.num_heads, dim_per_head).transpose(1, 3).reshape(-1, width, dim_per_head)
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = attn_weights / (dim_per_head ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.reshape(batch_size, self.num_heads, channel_num, width, dim_per_head).transpose(1, 2).reshape(batch_size, channel_num, width, dim)
        attn_output = self.post_attn_fc(attn_output)
        
        x = x + attn_output
        return x


class MHA_Test(nn.Module):
    def __init__(self, hidden_size):
        super(MHA_Test, self).__init__()
        self.filter_sizes = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.num_filters = 32
        self.num_heads = 8
        self.ffn_ratio = 3
        classifier_dropout = 0.1
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(3, self.num_filters, (k, 3), padding=(0, 1)) for k in self.filter_sizes])
        self.convs2 = nn.ModuleList(
            [nn.Conv2d(self.num_filters, self.num_filters, (k, hidden_size)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(classifier_dropout)
        num_classes = 2
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), num_classes)
        self.mha = MHA_Block(hidden_size)

    def conv_and_pool(self, x, conv1, conv2):
        x = conv1(x)
        x = self.mha(x)
        x = F.relu(conv2(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = x.float()
        # out = out.unsqueeze(1)
        hidden_state = torch.cat([self.conv_and_pool(out, self.convs1[i], self.convs2[i]) for i in range(len(self.filter_sizes))], 1)
        out = self.dropout(hidden_state)
        out = self.fc(out)
        return out, hidden_state


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=4):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MHA_SE_Block(nn.Module):
    def __init__(self, hidden_size, num_filters):
        super(MHA_SE_Block, self).__init__()
        self.num_heads = 8
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.post_attn_fc = nn.Linear(hidden_size, hidden_size)
        self.se = SE_Block(num_filters)
        
    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        batch_size, channel_num, width, dim = x.size()
        dim_per_head = dim // self.num_heads
        q = q.reshape(batch_size, channel_num, width, self.num_heads, dim_per_head).transpose(1, 3).reshape(-1, width, dim_per_head)
        k = k.reshape(batch_size, channel_num, width, self.num_heads, dim_per_head).transpose(1, 3).reshape(-1, width, dim_per_head)
        v = v.reshape(batch_size, channel_num, width, self.num_heads, dim_per_head).transpose(1, 3).reshape(-1, width, dim_per_head)
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = attn_weights / (dim_per_head ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.reshape(batch_size, self.num_heads, channel_num, width, dim_per_head).transpose(1, 2).reshape(batch_size, channel_num, width, dim)
        attn_output = self.post_attn_fc(attn_output)
        
        se_out = self.se(attn_output)
        output = se_out * attn_output
        x = x + output
        return x


class MHA_SE_Test(nn.Module):
    def __init__(self, hidden_size):
        super(MHA_SE_Test, self).__init__()
        self.filter_sizes = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.num_filters = 32
        self.num_heads = 8
        self.ffn_ratio = 3
        classifier_dropout = 0.1
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(3, self.num_filters, (k, 3), padding=(0, 1)) for k in self.filter_sizes])
        self.convs2 = nn.ModuleList(
            [nn.Conv2d(self.num_filters, self.num_filters, (k, hidden_size)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(classifier_dropout)
        num_classes = 2
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), num_classes)
        self.mha_se = MHA_SE_Block(hidden_size, self.num_filters)

    def conv_and_pool(self, x, conv1, conv2):
        x = conv1(x)
        x = self.mha_se(x)
        x = F.relu(conv2(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = x.float()
        # out = out.unsqueeze(1)
        hidden_state = torch.cat([self.conv_and_pool(out, self.convs1[i], self.convs2[i]) for i in range(len(self.filter_sizes))], 1)
        out = self.dropout(hidden_state)
        out = self.fc(out)
        return out, hidden_state


class Fca_Test(nn.Module):
    def __init__(self, hidden_size):
        super(Fca_Test, self).__init__()
        self.filter_sizes = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.num_filters = 32
        self.num_heads = 8
        self.ffn_ratio = 3
        classifier_dropout = 0.1
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(3, self.num_filters, (k, 3), padding=(0, 1)) for k in self.filter_sizes])
        self.convs2 = nn.ModuleList(
            [nn.Conv2d(self.num_filters, self.num_filters, (k, hidden_size)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(classifier_dropout)
        num_classes = 2
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), num_classes)
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(classifier_dropout)
        self.post_attn_fc = nn.Linear(hidden_size, hidden_size)
        self.post_attn_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn1 = nn.Linear(hidden_size, hidden_size * self.ffn_ratio)
        self.ffn2 = nn.Linear(hidden_size * self.ffn_ratio, hidden_size)
        self.ffn_dropout = nn.Dropout(classifier_dropout)
        self.post_ffn_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        
        self.fca = MultiSpectralAttentionLayer(self.num_filters, dct_h=56, dct_w=56, reduction=4, freq_sel_method="top16")

    def conv_and_pool(self, x, conv1, conv2):
        x = conv1(x)
        
        batch_size, channel_num, width, dim = x.size()
        dim_per_head = dim // self.num_heads
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q = q.reshape(batch_size, channel_num, width, self.num_heads, dim_per_head).transpose(1, 3).reshape(-1, width, dim_per_head)
        k = k.reshape(batch_size, channel_num, width, self.num_heads, dim_per_head).transpose(1, 3).reshape(-1, width, dim_per_head)
        v = v.reshape(batch_size, channel_num, width, self.num_heads, dim_per_head).transpose(1, 3).reshape(-1, width, dim_per_head)
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = attn_weights / (dim_per_head ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.reshape(batch_size, self.num_heads, channel_num, width, dim_per_head).transpose(1, 2).reshape(batch_size, channel_num, width, dim)
        attn_output = self.post_attn_fc(attn_output)
        
        fca_out = self.fca(attn_output)
        x = x + fca_out
        
        x = F.relu(conv2(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        
        return x

    def forward(self, x):
        out = x.float()
        # out = out.unsqueeze(1)
        hidden_state = torch.cat([self.conv_and_pool(out, self.convs1[i], self.convs2[i]) for i in range(len(self.filter_sizes))], 1)
        out = self.dropout(hidden_state)
        out = self.fc(out)
        return out, hidden_state


class MHAx2_Test(nn.Module):
    def __init__(self, hidden_size):
        super(MHAx2_Test, self).__init__()
        self.filter_sizes = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.num_filters = 32
        self.num_heads = 8
        self.ffn_ratio = 3
        classifier_dropout = 0.1
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(3, self.num_filters, (k, 3), padding=(0, 1)) for k in self.filter_sizes])
        self.convs2 = nn.ModuleList(
            [nn.Conv2d(self.num_filters, self.num_filters, (k, hidden_size)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(classifier_dropout)
        num_classes = 2
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), num_classes)
        self.mha1 = MHABlock(hidden_size)
        self.mha2 = MHABlock(hidden_size)

    def conv_and_pool(self, x, conv1, conv2):
        x = conv1(x)
        x = self.mha2(self.mha1(x))
        x = F.relu(conv2(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = x.float()
        # out = out.unsqueeze(1)
        hidden_state = torch.cat([self.conv_and_pool(out, self.convs1[i], self.convs2[i]) for i in range(len(self.filter_sizes))], 1)
        out = self.dropout(hidden_state)
        out = self.fc(out)
        return out, hidden_state