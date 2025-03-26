import torch
import torch.nn as nn
import math


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    return x*torch.sigmoid(x)


class ConvBlockOneStage(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlockOneStage, self).__init__()
        self.normalization = normalization
        self.ops = nn.ModuleList()
        self.conv = nn.Conv3d(n_filters_in, n_filters_out, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=0.1)

        if normalization == 'batchnorm':
            self.norm = nn.BatchNorm3d(n_filters_out)
        elif normalization == 'groupnorm':
            self.norm = nn.GroupNorm(num_groups=16, num_channels=n_filters_out)
        elif normalization == 'instancenorm':
            self.norm = nn.InstanceNorm3d(n_filters_out)
        elif normalization != 'none':
            assert False

    def forward(self, x):

        conv = self.conv(x)
        relu = self.relu(conv)
        if self.normalization != 'none':
            out = self.norm(relu)
        else:
            out = relu
        out = self.dropout(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        self.ops = nn.ModuleList()
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            self.ops.append(ConvBlockOneStage(input_channel, n_filters_out, normalization))

    def forward(self, x):

        for layer in self.ops:
            x = layer(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            ops.append(nn.ReLU(inplace=True))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            ops.append(nn.ReLU(inplace=True))

        self.ops = nn.Sequential(*ops)

    def forward(self, x):
        x = self.ops(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            ops.append(nn.ReLU(inplace=True))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            ops.append(nn.ReLU(inplace=True))

        self.ops = nn.Sequential(*ops)

    def forward(self, x):
        x = self.ops(x)
        return x


class Decoder(nn.Module):
    def __init__(self, n_classes=2, n_filters=16, normalization='none', has_dropout=False, dropout_rate=0.5):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=dropout_rate, inplace=True)


    def forward(self, x1, x2, x3, x4, x5):
        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)

        out = self.out_conv(x9)
        return out


class Encoder(nn.Module):
    def __init__(self, n_filters, normalization, has_dropout, dropout_rate):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.dropout = nn.Dropout3d(p=dropout_rate, inplace=True)

        self.latent_feature_in = nn.InstanceNorm3d(512, affine=True)

    def forward(self, input):
        x1 = input
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        latent_feature = x5
        if self.has_dropout:
            x5 = self.dropout(x5)

        return x1, x2, x3, x4, x5, self.latent_feature_in(latent_feature)




class TembFusion(nn.Module):
    def __init__(self, n_filters_out):
        super(TembFusion, self).__init__()
        self.temb_proj = torch.nn.Linear(512, n_filters_out)
    def forward(self, x, temb):
        if temb is not None:
            x = x + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]
        else:
            x =x
        return x



class TimeStepEmbedding(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, normalization='none'):
        super(TimeStepEmbedding, self).__init__()

        self.embed_dim_in = 128
        self.embed_dim_out = self.embed_dim_in * 4

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.embed_dim_in,
                            self.embed_dim_out),
            torch.nn.Linear(self.embed_dim_out,
                            self.embed_dim_out),
        ])

        self.emb = ConvBlockTemb(1, n_filters_in, n_filters_out, normalization=normalization)

    def forward(self, x, t=None, image=None):
        if t is not None:
            temb = get_timestep_embedding(t, self.embed_dim_in)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)

        else:
            temb = None


        if image is not None :
            x = torch.cat([image, x], dim=1)
        x = self.emb(x, temb)

        return x, temb






class ConvBlockTembOneStage(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, normalization='none', temb_flag=True):
        super(ConvBlockTembOneStage, self).__init__()

        self.temb_flag = temb_flag
        self.normalization = normalization

        self.ops = nn.ModuleList()

        self.conv = nn.Conv3d(n_filters_in, n_filters_out, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=0.1)


        if normalization == 'batchnorm':
            self.norm = nn.BatchNorm3d(n_filters_out)
        elif normalization == 'groupnorm':
            self.norm = nn.GroupNorm(num_groups=16, num_channels=n_filters_out)
        elif normalization == 'instancenorm':
            self.norm = nn.InstanceNorm3d(n_filters_out)
        elif normalization != 'none':
            assert False

        self.temb_fusion =  TembFusion(n_filters_out)


    def forward(self, x, temb):

        conv = self.conv(x)
        relu = self.relu(conv)
        if self.normalization != 'none':
            norm = self.norm(relu)
        else:
            norm = relu
        norm = self.dropout(norm)
        if self.temb_flag:
            out = self.temb_fusion(norm, temb)
        else:
            out = norm
        return out


class ConvBlockTemb(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlockTemb, self).__init__()

        self.ops = nn.ModuleList()
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            if i < n_stages-1:
                self.ops.append(ConvBlockTembOneStage(input_channel, n_filters_out, normalization))
            else:
                self.ops.append(ConvBlockTembOneStage(input_channel, n_filters_out, normalization, temb_flag=False))

    def forward(self, x, temb):

        for layer in self.ops:
            x = layer(x, temb)
        return x



class network(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, dropout_rate=0.5):
        super(network, self).__init__()
        self.has_dropout = has_dropout
        self.n_classes = n_classes
        self.time_steps = 1000

        self.embedding = TimeStepEmbedding(n_channels, n_filters, normalization=normalization)

        self.encoder_norm = Encoder(n_filters, normalization, has_dropout, dropout_rate)
        self.decoder_xi = Decoder(n_classes, n_filters, normalization, has_dropout, dropout_rate)

    def forward(self, image=None):
        x, temb = self.embedding(image)
        x1, x2, x3, x4, x5, latent_feature_temp = self.encoder_norm(x)
        return self.decoder_xi(x1, x2, x3, x4, x5), latent_feature_temp

