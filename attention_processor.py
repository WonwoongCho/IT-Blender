import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple, Union

class FluxBlendedAttnProcessor2_0(nn.Module):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, hidden_dim, ba_scale=1.0, num_ref=1, temperature=1.2):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxBlendedAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        
        self.blended_attention_k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.blended_attention_v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ba_scale = ba_scale
        self.num_ref = num_ref
        self.temperature = temperature # this is used only when num_ref > 1

    def __call__(
        self,
        attn, #: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        is_negative_prompt: bool = False
    ) -> torch.FloatTensor:
        assert encoder_hidden_states is None, "It should be given as None because we are applying it-blender only to the single streams."
        batch_size, _, _ = hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            normalized_query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(normalized_query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)


        # [noisy, clean]
        chunk = batch_size//(1+self.num_ref)
        ba_query = normalized_query[:chunk]  # noisy query

        ba_key = self.blended_attention_k_proj(hidden_states[chunk:]) # clean key
        ba_value = self.blended_attention_v_proj(hidden_states[chunk:]) # clean value

        ba_key = ba_key.view(chunk, -1, attn.heads, head_dim).transpose(1, 2) # the -1 is gonna be multiplied by self.num_ref
        ba_value = ba_value.view(chunk, -1, attn.heads, head_dim).transpose(1, 2)

        ba_hidden_states = F.scaled_dot_product_attention(
            ba_query, ba_key, ba_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, scale=(1 / math.sqrt(ba_query.size(-1)))*self.temperature if self.num_ref > 1 else 1 / math.sqrt(ba_query.size(-1))
        )

        ba_hidden_states = ba_hidden_states.transpose(1, 2).reshape(chunk, -1, attn.heads * head_dim)
        ba_hidden_states = ba_hidden_states.to(query.dtype)

        zero_tensor_list = [torch.zeros_like(ba_hidden_states)]*self.num_ref
        ba_hidden_states = torch.cat([ba_hidden_states]+zero_tensor_list, dim=0)    
            
        hidden_states = hidden_states + self.ba_scale * ba_hidden_states
        
        return hidden_states


# For SD 1.5
class BlendedAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, scale=1.0):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("BlendedAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        SD_15_sa_name_list = ['down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor', 'down_blocks.0.attentions.1.transformer_blocks.0.attn1.processor', 'down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor', 'down_blocks.1.attentions.1.transformer_blocks.0.attn1.processor', 'down_blocks.2.attentions.0.transformer_blocks.0.attn1.processor', 'down_blocks.2.attentions.1.transformer_blocks.0.attn1.processor', 'mid_block.attentions.0.transformer_blocks.0.attn1.processor', 'up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor', 'up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor', 'up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor', 'up_blocks.2.attentions.0.transformer_blocks.0.attn1.processor', 'up_blocks.2.attentions.1.transformer_blocks.0.attn1.processor', 'up_blocks.2.attentions.2.transformer_blocks.0.attn1.processor', 'up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor', 'up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor', 'up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor']

        self.hidden_size = hidden_size
        self.scale = scale

        self.to_k_blended_attention = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v_blended_attention = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)


        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # for it-blender We leverage the key and value from the clean image.
        
        img_emb = hidden_states[batch_size//2:] # our proposed idea
        
        ba_key = self.to_k_blended_attention(img_emb)
        ba_value = self.to_v_blended_attention(img_emb)

        ba_key = ba_key.view(batch_size//2, -1, attn.heads, head_dim).transpose(1, 2)
        ba_value = ba_value.view(batch_size//2, -1, attn.heads, head_dim).transpose(1, 2)
        
        ba_hidden_states = F.scaled_dot_product_attention(
            query[:batch_size//2], ba_key, ba_value, attn_mask=None, dropout_p=0.0, is_causal=False 
        )
    
        ba_hidden_states = ba_hidden_states.transpose(1, 2).reshape(batch_size//2, -1, attn.heads * head_dim)
        ba_hidden_states = ba_hidden_states.to(query.dtype)
        
        # for sampling
        num_samples = batch_size//4 # [noisy uncond, noisy cond, clean uncond, clean cond]
        ba_hidden_states = torch.cat([torch.zeros_like(ba_hidden_states[:num_samples]), ba_hidden_states[num_samples:num_samples*2], torch.zeros_like(ba_hidden_states)], dim=0)
        
        if attention_mask is not None: # [noisy uncond, noisy cond, clean uncond, clean cond] -> [0, mask, 0, 0]
            attention_mask = F.interpolate(
                attention_mask,
                size=int(math.sqrt(ba_hidden_states.size(1)))
            )
            attention_mask = attention_mask.reshape(attention_mask.size(0), attention_mask.size(1), -1)
            attention_mask = attention_mask.permute(0,2,1)
            ba_hidden_states *= torch.cat([torch.zeros_like(attention_mask), attention_mask, torch.zeros_like(attention_mask), torch.zeros_like(attention_mask)], dim=0)
        
        hidden_states = hidden_states + self.scale * ba_hidden_states
                    
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        is_training=True,
    ):
        super().__init__()
        self.is_training = is_training

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1            

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        with torch.no_grad():
            if self.is_training:
                # for training
                self.attn_map = query[:batch_size//2] @ key.transpose(-2, -1).softmax(dim=-1)[:batch_size//2]
            else:
                # for sampling,  
                num_samples = batch_size // 4
                # noisy (uncond, cond), clean (uncond, cond), and we wanna get the noisy cond map
                # Key here is text.  Key[num_samples:num_samples*2] is the cond text.
                self.attn_map = query[num_samples:num_samples*2] @ key.transpose(-2, -1).softmax(dim=-1)[num_samples:num_samples*2]

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
