import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from fairseq import options, utils

from . import (
    FairseqEncoder, FairseqIncrementalDecoder, FairseqModel, register_model,
    register_model_architecture
)
from .lstm import LSTMEncoder, LSTMDecoder, LSTMModel, Linear

@register_model('edit-lstm')
class EditLSTMModel(LSTMModel):
    def forward(self, src_tokens, src_lengths, src_insert, prev_output_tokens):
        encoder_out = self.encoder(src_tokens, src_lengths, src_insert)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        return decoder_out

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        pretrained_encoder_embed = None
        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim)
        pretrained_decoder_embed = None
        if args.decoder_embed_path:
            pretrained_decoder_embed = load_pretrained_embedding_from_file(
                args.decoder_embed_path, task.target_dictionary, args.decoder_embed_dim)

        additional_input_size = 0

        encoder = EditLSTMEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
        )

        decoder = EditLSTMDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=options.eval_bool(args.decoder_attention),
            encoder_embed_dim=args.encoder_embed_dim,
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            additional_input_size=additional_input_size,
        )

        return cls(encoder, decoder)


class EditLSTMEncoder(LSTMEncoder):
    #def __init__(
    #    self, dictionary, embed_dim=512, hidden_size=512, num_layers=1,
    #    dropout_in=0.1, dropout_out=0.1, bidirectional=False,
    #    left_pad=True, pretrained_embed=None, padding_value=0.,
    #):
    #    super().__init__(
    #        dictionary, embed_dim=embed_dim, hidden_size=hidden_size, num_layers=num_layers,
    #        dropout_in=dropout_in, dropout_out=dropout_out, bidirectional=bidirectional,
    #        left_pad=left_pad, pretrained_embed=pretrained_embed, padding_value=padding_value,
    #        )

    def forward(self, src_tokens, src_lengths, src_insert):
        encoder_out_dict = super().forward(src_tokens, src_lengths)
        # x: (T, B, C)  final_*: (1, B, C)
        #x, final_hiddens, final_cells = encoder_out_dict['encoder_out']
        insert_embedding = self.embed_tokens(src_insert)  # (B, T=1, C)
        insert_embedding = insert_embedding.transpose(0, 1)  # B x T x C -> T x B x C
        encoder_out_dict['encoder_insert'] = insert_embedding
        return encoder_out_dict

    def reorder_encoder_out(self, encoder_out_dict, new_order):
        super().reorder_encoder_out(encoder_out_dict, new_order)
        encoder_out_dict['encoder_insert'] = \
            encoder_out_dict['encoder_insert'].index_select(1, new_order)
        return encoder_out_dict


class EditLSTMDecoder(LSTMDecoder):
    """LSTM decoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_embed_dim=512, encoder_output_units=512, pretrained_embed=None,
        additional_input_size=0,
    ):
        self.use_fc = False
        if self.use_fc:
            self.fc_insert = Linear(encoder_output_units + encoder_embed_dim, hidden_size)
        else:
            hidden_size = hidden_size + encoder_embed_dim
            out_embed_dim = out_embed_dim + encoder_embed_dim

        super().__init__(
            dictionary, embed_dim=embed_dim, hidden_size=hidden_size, out_embed_dim=out_embed_dim,
            num_layers=num_layers, dropout_in=dropout_in, dropout_out=dropout_out, attention=attention,
            encoder_embed_dim=encoder_embed_dim, encoder_output_units=encoder_output_units, pretrained_embed=pretrained_embed,
            additional_input_size=additional_input_size,
        )

    def concat_with_state(self, state, t):
        # state: (L, B, C)
        # t: (1, B, C')
        num_layers = state.size(0)
        t = t.repeat(num_layers, 1, 1)  # (L, B, C')
        new_state = torch.cat((state, t), dim=2)
        return new_state

    def forward(self, prev_output_tokens, encoder_out_dict, incremental_state=None, augmenting_inputs=None):
        encoder_out = encoder_out_dict['encoder_out']
        encoder_padding_mask = encoder_out_dict['encoder_padding_mask']
        encoder_insert = encoder_out_dict['encoder_insert']

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, _, _ = encoder_out[:3]
        srclen = encoder_outs.size(0)

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # concat augmenting inputs
        if augmenting_inputs is not None:
            x = torch.cat([x, augmenting_inputs], 2)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        else:
            _, encoder_hiddens, encoder_cells = encoder_out[:3]

            # Combine with insertions
            encoder_hiddens = self.concat_with_state(encoder_hiddens, encoder_insert)
            encoder_cells = self.concat_with_state(encoder_cells, encoder_insert)
            if self.use_fc:
                encoder_hiddens = self.fc_insert(encoder_hiddens)
                encoder_cells = self.fc_insert(encoder_cells)

            num_layers = len(self.layers)
            prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
            prev_cells = [encoder_cells[i] for i in range(num_layers)]
            input_feed = x.data.new(bsz, self.hidden_size).zero_()

        attn_scores = x.data.new(srclen, seqlen, bsz).zero_()
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = torch.cat((x[j, :, :], input_feed), dim=1)
            #print(input_feed.size())
            #print(input.size())

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs, encoder_padding_mask)
            else:
                out = hidden
            out = F.dropout(out, p=self.dropout_out, training=self.training)

            # input feeding
            input_feed = out

            # save final output
            outs.append(out)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state', (prev_hiddens, prev_cells, input_feed))

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        attn_scores = attn_scores.transpose(0, 2)

        # project back to size of vocabulary
        if hasattr(self, 'additional_fc'):
            x = self.additional_fc(x)
            x = F.dropout(x, p=self.dropout_out, training=self.training)
        outputs = x  # last layer representation
        x = self.fc_out(x)  # logits

        return x, attn_scores, outputs


@register_model_architecture('edit-lstm', 'edit-lstm')
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', args.encoder_embed_dim)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', False)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', args.dropout)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', args.dropout)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', args.decoder_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    args.decoder_attention = getattr(args, 'decoder_attention', '1')
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', args.dropout)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', args.dropout)

