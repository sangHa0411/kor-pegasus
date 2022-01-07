import math
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from typing import Optional

from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.pegasus.configuration_pegasus import PegasusConfig

from transformers.models.pegasus.modeling_pegasus import (
    PegasusForConditionalGeneration,
    PegasusModel, 
    PegasusSinusoidalPositionalEmbedding,
    PegasusEncoder, 
    PegasusDecoder,
    PegasusEncoderLayer,
    PegasusDecoderLayer
)

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class PegasusAlbertEncoder(PegasusEncoder):
    def __init__(self, config: PegasusConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = PegasusSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
            self.padding_idx,
        )

        shared_layer = PegasusEncoderLayer(config)
        self.layers = nn.ModuleList([shared_layer for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False

class PegasusAlbertDecoder(PegasusDecoder):
    def __init__(self, config: PegasusConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = PegasusSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            self.padding_idx,
        )

        shared_layer = PegasusDecoderLayer(config)
        self.layers = nn.ModuleList([shared_layer for _ in range(config.decoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False


class PegasusAlbertModel(PegasusModel):
    def __init__(self, config: PegasusConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = PegasusAlbertEncoder(config, self.shared)
        self.decoder = PegasusAlbertDecoder(config, self.shared)


class PegasusForPretraining(PegasusForConditionalGeneration):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
        r"embed_positions\.weight",
    ]

    def __init__(self, config: PegasusConfig):
        super().__init__(config)
        self.model = PegasusAlbertModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        mlm_labels=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs.last_hidden_state
        encoder_last_hidden_state = outputs.encoder_last_hidden_state

        gsc_logits = self.lm_head(last_hidden_state) + self.final_logits_bias
        mlm_logits = self.lm_head(encoder_last_hidden_state) + self.final_logits_bias 

        gsc_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            gsc_loss = loss_fct(gsc_logits.view(-1, self.config.vocab_size), labels.view(-1))
            
            if mlm_labels is not None :
                mlm_loss = loss_fct(mlm_logits.view(-1, self.config.vocab_size), mlm_labels.view(-1))
                gsc_loss = gsc_loss + mlm_loss

        if not return_dict:
            output = (gsc_logits,) + outputs[1:]
            return ((gsc_loss,) + output) if gsc_loss is not None else output

        return Seq2SeqLMOutput(
            loss=gsc_loss,
            logits=gsc_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
