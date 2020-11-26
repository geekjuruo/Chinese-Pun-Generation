# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from fairseq import options, utils
from fairseq.modules import AdaptiveSoftmax
from . import (
    FairseqEncoder, FairseqIncrementalDecoder, FairseqModel, register_model,
    FairseqLanguageModel, register_model_architecture
)


@register_model('lstm')
class LSTMModel(FairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder', default='lstm', choices=['lstm', 'bow'])
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-hidden-size', type=int, metavar='N',
                            help='encoder hidden size')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='number of encoder layers')
        parser.add_argument('--encoder-bidirectional', action='store_true',
                            help='make all layers of encoder bidirectional')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
                            help='decoder hidden size')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='number of decoder layers')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=str, metavar='BOOL',
                            help='decoder attention')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument('--encoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for encoder input embedding')
        parser.add_argument('--encoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for encoder output')
        parser.add_argument('--decoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for decoder output')
        parser.add_argument('--share-decoder-input-output-embed', default=False,
                            action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', default=False, action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')

        parser.add_argument('--pretrained-lm', default=None, type=str, metavar='STR',
                            help='path to pre-trained language model')
        parser.add_argument('--fusion-type', default='output', choices=['output', 'input', 'prob'],
                            help='where to fuse pretrained models')
        parser.add_argument('--mixing-weights', default=None, type=str,
                            help='mixing weights (probability) of the models')

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

        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim)
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(
                num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
            )

        if args.share_all_embeddings:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise ValueError('--share-all-embeddings requires a joint dictionary')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError(
                    '--share-all-embed not compatible with --decoder-embed-path'
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to '
                    'match --decoder-embed-dim'
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            if args.decoder_embed_path:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    args.decoder_embed_path,
                    task.target_dictionary,
                    args.decoder_embed_dim
                )
        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
                args.decoder_embed_dim != args.decoder_out_embed_dim):
            raise ValueError(
                '--share-decoder-input-output-embeddings requires '
                '--decoder-embed-dim to match --decoder-out-embed-dim'
            )

        additional_input_size = 0
        if args.pretrained_lm:
            lm_args = copy.copy(args)
            setattr(lm_args, 'task', 'language_modeling')
            # For loading vocab
            setattr(lm_args, 'data', os.path.dirname(args.pretrained_lm))
            lm_task = tasks.setup_task(lm_args)
            print('| loading pretrained LM from {}'.format(args.pretrained_lm))
            lm = utils.load_ensemble_for_inference([args.pretrained_lm], lm_task)[0][0]
            if args.fusion_type == 'input':
                additional_input_size = lm.decoder.output_size

        #if args.encoder == 'lstm':
        encoder = LSTMEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
        )
        #elif args.encoder == 'bow':
        #    encoder = BoWEncoder(
        #        dictionary=task.source_dictionary,
        #        embed_dim=args.encoder_embed_dim,
        #        dropout_in=args.encoder_dropout_in,
        #        pretrained_embed=pretrained_encoder_embed,
        #        )

        decoder = LSTMDecoder(
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
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
        )

        if args.pretrained_lm:
            decoders = [decoder, lm.decoder]
            trainable = [True, False]
            conditional = [True, False]

            if args.fusion_type == 'input':
                for decoder in decoders[1:]:
                    decoder.disable_output_layer()
            elif args.fusion_type == 'output':
                for decoder in decoders:
                    decoder.disable_output_layer()

            if args.fusion_type == 'prob':
                if args.mixing_weights is not None and args.mixing_weights != 'learned':
                    mixing_weights = eval(args.mixing_weights)
                    assert isinstance(mixing_weights, list)
                    assert sum(mixing_weights) == 1, 'Mixing weights do not sum to 1: {}'.format(mixing_weights)
                else:
                    mixing_weights = args.mixing_weights
            else:
                mixing_weights = None

            decoder = MultiDecoder(decoders, trainable, conditional,
                    fusion_type=args.fusion_type, mixing_weights=mixing_weights)

        return cls(encoder, decoder)


@register_model('lstm_lm')
class LSTMLanguageModel(FairseqLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
                            help='decoder hidden size')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='number of decoder layers')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--decoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for decoder output')
        parser.add_argument('--share-decoder-input-output-embed', default=False,
                            action='store_true',
                            help='share decoder input and output embeddings')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_lm_architecture(args)

        if hasattr(args, 'max_target_positions'):
            args.tokens_per_sample = args.max_target_positions

        decoder = LSTMDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=options.eval_bool(args.decoder_attention),
            encoder_embed_dim=args.decoder_embed_dim,
            encoder_output_units=args.decoder_hidden_size,
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
        )
        return LSTMLanguageModel(decoder)


class LSTMEncoder(FairseqEncoder):
    """LSTM encoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, num_layers=1,
        dropout_in=0.1, dropout_out=0.1, bidirectional=False,
        left_pad=True, pretrained_embed=None, padding_value=0.,
    ):
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2


    def forward(self, src_tokens, src_lengths):
        if self.left_pad:
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                self.padding_idx,
                left_to_right=True,
            )

        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.data.new(*state_size).zero_()
        c0 = x.data.new(*state_size).zero_()
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_value)
        x = F.dropout(x, p=self.dropout_out, training=self.training)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:

            def combine_bidir(outs):
                return outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous().view(self.num_layers, bsz, -1)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        return {
            'encoder_out': (x, final_hiddens, final_cells),
            'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = tuple(
            eo.index_select(1, new_order)
            for eo in encoder_out['encoder_out']
        )
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


class BoWEncoder(LSTMEncoder):
    def __init__(
        self, dictionary, embed_dim=512, dropout_in=0.1,
        left_pad=True, pretrained_embed=None, padding_value=0.,
    ):
        FairseqEncoder.__init__(self, dictionary)
        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed
        self.padding_value = padding_value
        self.output_units = embed_dim
        self.dropout_in = dropout_in

    def forward(self, src_tokens, src_lengths):
        # embed tokens
        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()  # T x B

        # non-padding elements
        m = encoder_padding_mask.eq(0).float()  # T x B
        mean_x = torch.sum(x * m.unsqueeze(2), 0) / torch.sum(m, 0).unsqueeze(1)
        mean_x = mean_x.unsqueeze(0)
        # TODO: mask pad
        #mean_x = x.mean(dim=0, keepdim=True)  # B x C

        return {
            'encoder_out': (x, mean_x, mean_x),
            'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None
        }


class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, output_embed_dim):
        # input_embed_dim: decoder hidden_size
        # output_embed_dim: encoder_out_units
        super().__init__()

        self.input_proj = Linear(input_embed_dim, output_embed_dim, bias=False)
        self.output_proj = Linear(input_embed_dim + output_embed_dim, output_embed_dim, bias=False)

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x output_embed_dim

        # x: bsz x output_embed_dim
        x = self.input_proj(input)

        # compute attention
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)

        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)

        x = F.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores

class MultiDecoder(FairseqIncrementalDecoder):
    def __init__(self, decoders, trainable, conditional, fusion_type='prob', mixing_weights=None):
        assert len(decoders) == len(trainable) and \
                len(decoders) == len(conditional)

        # Assume decoders[0] is the base / in-domain decoder
        super().__init__(decoders[0].dictionary)
        self.decoders = nn.ModuleList(decoders)
        self.num_decoders = len(decoders)
        self.conditional = conditional
        self.fusion_type = fusion_type

        self.mixing_weights = None
        if fusion_type == 'prob':
            if mixing_weights is None:
                mixing_weights = [1. / self.num_decoders] * self.num_decoders
            self.mixing_weights = mixing_weights
            print('| decoder mixing weights: {}'.format(self.mixing_weights))
            if self.mixing_weights == 'learned':
                base_decoder = self.decoders[0]
                # Input: base_decoder output (B, T, C)
                # Output: mixing weights at each time step (B, T, num_decoders)
                self.mixing_gate = nn.Sequential(nn.Linear(base_decoder.hidden_size, self.num_decoders), nn.Softmax(dim=2))
        elif fusion_type == 'output':
            concat_size = sum([decoder.output_size for decoder in self.decoders])
            self.fusion_fc_out = nn.Linear(concat_size, len(self.dictionary))

        # Map word embedding. Assuming that the first decoder is the base decoder.
        dict_maps = [None]
        tgt_dict = self.dictionary  # decoders[0].dictionary
        for decoder in self.decoders[1:]:
            dict_map = self.get_dict_map(decoder.dictionary, tgt_dict)
            dict_maps.append(dict_map)
            # Map word embeddings to base decoder dictionary
            new_weight = decoder.embed_tokens.weight[dict_map, :]
            decoder.embed_tokens.weight = Parameter(new_weight)
        self.dict_maps = dict_maps

        for i, decoder in enumerate(self.decoders):
            if not trainable[i]:
                for name, param in decoder.named_parameters():
                    param.requires_grad = False

    def get_dict_map(self, src_dict, tgt_dict):
        dict_map = []
        for idx in range(len(tgt_dict)):
            dict_map.append(src_dict.index(tgt_dict[idx]))
        return dict_map

    def get_init_incremental_state(self):
        return [d.get_init_incremental_state() for d in self.decoders]

    def forward(self, prev_output_tokens, encoder_out, incremental_states=None):
        if incremental_states is not None:
            assert len(incremental_states) == self.num_decoders
        else:
            incremental_states = [None] * self.num_decoders

        # Compute decoder outputs of all non-base / augmenting decoders
        all_decoder_outs = []
        for i, (decoder, conditional, incremental_state) in enumerate(
                zip(self.decoders, self.conditional, incremental_states)):
            if i == 0:
                continue
            _encoder_out = None if not conditional else encoder_out
            decoder_out = decoder(prev_output_tokens, _encoder_out, incremental_state)
            all_decoder_outs.append(decoder_out)

        if self.fusion_type == 'input':
            augmenting_inputs = torch.cat([decoder_out[2] for decoder_out in all_decoder_outs], 2)
            decoder_out = self.decoders[0](prev_output_tokens, encoder_out, incremental_states[0], augmenting_inputs=augmenting_inputs)
            return decoder_out
        else:
            # Run base decoder
            decoder_out = self.decoders[0](prev_output_tokens, encoder_out, incremental_states[0])
            all_decoder_outs.insert(0, decoder_out)
            attn = decoder_out[1]

            if self.fusion_type == 'prob':
                if self.mixing_weights == 'learned':
                    x = all_decoder_outs[0][2]  # base decoder outputs (B, T, C)
                    mixing_weights = self.mixing_gate(x)  # (B, T, D)

                all_probs = []
                for i, (decoder, decoder_out, dict_map) in enumerate(
                        zip(self.decoders, all_decoder_outs, self.dict_maps)):
                    probs = decoder.get_normalized_probs(decoder_out, False, None)  # (B, T, V)
                    if dict_map:
                        probs = probs[:, :, dict_map]
                    if self.mixing_weights == 'learned':
                        probs = probs * mixing_weights[:, :, i].unsqueeze(2)
                    else:
                        probs = probs * self.mixing_weights[i]
                    all_probs.append(probs)

                final_probs = sum(all_probs)
                return final_probs, attn

            elif self.fusion_type == 'output':
                outputs = torch.cat([decoder_out[2] for decoder_out in all_decoder_outs], 2)
                # Map to vocab size
                x = self.fusion_fc_out(outputs)
                return x, attn

    def reorder_incremental_state(self, incremental_states, new_order):
        for decoder, incremental_state in zip(self.decoders, incremental_states):
            decoder.reorder_incremental_state(incremental_state, new_order)

    def get_normalized_probs(self, net_output, log_probs, _):
        """Get normalized probabilities (or log probs) from a net's output."""
        if self.fusion_type == 'prob':
            if not log_probs:
                return net_output[0]
            else:
                return torch.log(net_output[0])
        else:
            return super().get_normalized_probs(net_output, log_probs, _)

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return min([d.max_positions() for d in self.decoders])


class LSTMDecoder(FairseqIncrementalDecoder):
    """LSTM decoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_embed_dim=512, encoder_output_units=512, pretrained_embed=None,
        additional_input_size=0,
        share_input_output_embed=False, adaptive_softmax_cutoff=None,
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units
        #assert encoder_output_units == hidden_size, \
        #    'encoder_output_units ({}) != hidden_size ({})'.format(encoder_output_units, hidden_size)
        # TODO another Linear layer if not equal

        # input_size = attention_output_size + input_embed_size
        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=hidden_size + embed_dim + additional_input_size if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])
        self.attention = AttentionLayer(hidden_size, encoder_output_units) if attention else None
        if hidden_size != out_embed_dim:
            self.additional_fc = Linear(hidden_size, out_embed_dim)
        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(num_embeddings, embed_dim, adaptive_softmax_cutoff,
                                                    dropout=dropout_out)
        elif not self.share_input_output_embed:
            self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)


    @property
    def output_size(self):
        return self.hidden_size

    def disable_output_layer(self):
        for param in self.fc_out.parameters():
            param.requires_grad = False

    def forward(self, prev_output_tokens, encoder_out_dict, incremental_state=None, augmenting_inputs=None):
        encoder_out = encoder_out_dict['encoder_out']
        encoder_padding_mask = encoder_out_dict['encoder_padding_mask']

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
            num_layers = len(self.layers)
            prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
            prev_cells = [encoder_cells[i] for i in range(num_layers)]
            input_feed = x.data.new(bsz, self.hidden_size).zero_()

        attn_scores = x.data.new(srclen, seqlen, bsz).zero_()
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = torch.cat((x[j, :, :], input_feed), dim=1)

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
        if not self.training and self.need_attn:
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None

        # project back to size of vocabulary
        if self.adaptive_softmax is None:
            if hasattr(self, 'additional_fc'):
                x = self.additional_fc(x)
                x = F.dropout(x, p=self.dropout_out, training=self.training)
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = self.fc_out(x)
        return x, attn_scores

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


@register_model_architecture('lstm', 'lstm')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
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
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', '10000,50000,200000')

@register_model_architecture('lstm_lm', 'lstm_lm')
def base_lm_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', args.decoder_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    args.decoder_attention = getattr(args, 'decoder_attention', '1')
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', args.dropout)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', args.dropout)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', '10000,50000,200000')

@register_model_architecture('lstm', 'lstm_wiseman_iwslt_de_en')
def lstm_wiseman_iwslt_de_en(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', 0)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', 0)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', args.dropout)
    base_architecture(args)


@register_model_architecture('lstm', 'lstm_luong_wmt_en_de')
def lstm_luong_wmt_en_de(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1000)
    args.encoder_layers = getattr(args, 'encoder_layers', 4)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1000)
    args.decoder_layers = getattr(args, 'decoder_layers', 4)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 1000)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', 0)
    base_architecture(args)
