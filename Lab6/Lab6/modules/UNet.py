import torch
from torch import nn
from modules.layers import MyConvo2d, SpatialTransformer, ResidualBlock, TimestepBlock

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x

class UNet(nn.Module):
    def __init__(
            self, 
            image_size, 
            in_channels, 
            model_channels, 
            out_channels, 
            num_res_blocks, 
            n_heads, 
            transformer_n_layers, 
            attention_resolutions=[2,4,8], 
            dropout=0, 
            channel_mult=[1,2,4,5], 
            num_classes=24,
            n_embd=8192, # for vq model
            context_dim=128, # condition
        ):
        super(UNet, self).__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_resolutions = len(channel_mult)

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.label_emb = nn.Linear(num_classes, context_dim)

        # Input Blocks
        self.Encoder = nn.ModuleList([TimestepEmbedSequential(MyConvo2d(in_channels, model_channels, 3))])
        ch = model_channels
        ds = 1
        input_chs = []
        for level, mult in enumerate(self.channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(input_dim=ch, emb_dim=time_embed_dim, output_dim=mult * model_channels, kernel_size=3)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(SpatialTransformer(ch, n_heads=n_heads, n_layers=transformer_n_layers, dropout=dropout, context_dim=context_dim))
                self.Encoder.append(TimestepEmbedSequential(*layers))
                input_chs.append(ch)
            if level != len(self.channel_mult) - 1:
                out_ch = ch
                self.Encoder.append(TimestepEmbedSequential(ResidualBlock(input_dim=ch, emb_dim=time_embed_dim, output_dim=out_ch, kernel_size=3)))
                ds *= 2
                ch = out_ch
                input_chs.append(ch)

        # Middle Blocks
        self.middle = TimestepEmbedSequential(
            ResidualBlock(input_dim=ch, emb_dim=time_embed_dim, output_dim=ch, kernel_size=3),
            SpatialTransformer(ch, n_heads=n_heads, n_layers=transformer_n_layers, dropout=dropout, context_dim=context_dim),
            ResidualBlock(input_dim=ch, emb_dim=time_embed_dim, output_dim=ch, kernel_size=3),
        )

        # Output Blocks
        self.Decoder = nn.ModuleList([])
        for level, mult in reversed(list(enumerate(self.channel_mult))):
            for i in range(num_res_blocks + 1):
                if len(input_chs) == 0:
                    break
                ich = input_chs.pop()
                layers = [ResidualBlock(input_dim=ch + ich, emb_dim=time_embed_dim, output_dim=mult * model_channels, kernel_size=3)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(SpatialTransformer(ch, n_heads=n_heads, n_layers=transformer_n_layers, dropout=dropout, context_dim=context_dim))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(ResidualBlock(input_dim=ch, emb_dim=time_embed_dim, output_dim=out_ch, kernel_size=3))
                    ds //= 2
                self.Decoder.append(TimestepEmbedSequential(*layers))
        out_ch = ch
        self.out = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            MyConvo2d(out_ch, out_channels, 1),
        )

    def forward(self, x, time_step, context):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        """
        hs = []
        time_step = time_step.to(torch.float32).view(-1, 1)
        emb = self.time_embed(time_step)
        context = self.label_emb(context)
        for module in self.Encoder:
            x = module(x, emb, context)
            hs.append(x)
        x = self.middle(x, emb, context)
        for module in self.Decoder:
            x = module(torch.cat([x, hs.pop()], dim=1), emb, context)
        return self.out(x)