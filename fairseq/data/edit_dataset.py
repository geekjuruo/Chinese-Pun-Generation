import numpy as np
import torch

from fairseq import utils

from . import data_utils, LanguagePairDataset, IndexedInMemoryDataset
from .dictionary import Dictionary


class EditDictionary(Dictionary):
    def __init__(self, pad='<pad>', eos='</s>', unk='<unk>'):
        self.unk_word, self.pad_word, self.eos_word = unk, pad, eos
        self.soi_word, self.eoi_word = '<i>', '</i>'
        self.placeholder_word = '<placeholder>'
        self.symbols = []
        self.count = []
        self.indices = {}
        # dictionary indexing starts at 1 for consistency with Lua
        self.add_symbol('<Lua heritage>')
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        # start/end of insertion
        self.soi_index = self.add_symbol(self.soi_word)
        self.eoi_index = self.add_symbol(self.eoi_word)
        self.placeholder_index = self.add_symbol(self.placeholder_word)
        self.nspecial = len(self.symbols)

class EditDatasetSrcWrapper(object):
    def __init__(self, ds):
        self.ds = ds
        self.size = ds.size // 3
        self.sizes = np.array([ds.sizes[i] for i in range(2, len(ds), 3)])

    def __getitem__(self, index):
        return {
                'deleted': self.ds[index * 2 + 0],
                'template': self.ds[index * 2 + 1],
                }

    def __len__(self):
        return len(self.ds) // 3

def collate(samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False, insert='none'):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source-template', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source-template'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    src_insert = None
    if insert != 'none':
        src_insert = merge('source-insert', left_pad=left_pad_source)
        src_insert = src_insert.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        # we create a shifted version of targets for feeding the
        # previous output token(s) into the next decoder step
        prev_output_tokens = merge(
            'target',
            left_pad=left_pad_target,
            move_eos_to_beginning=True,
        )
        prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)
    else:
        ntokens = sum(len(s['source-template']) for s in samples)

    item = {
        'id': id,
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'prev_output_tokens': prev_output_tokens,
        },
        'target': target,
    }
    if insert != 'none':
        item['net_input']['src_insert'] = src_insert
    return item



class EditDataset(LanguagePairDataset):
    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True,
        insert='none', combine='embedding',
    ):
        super().__init__(
            src, src_sizes, src_dict,
            tgt=tgt, tgt_sizes=tgt_sizes, tgt_dict=tgt_dict,
            left_pad_source=left_pad_source, left_pad_target=left_pad_target,
            max_source_positions=max_source_positions, max_target_positions=max_target_positions,
            shuffle=shuffle,
            )
        self.insert = insert
        self.combine = combine

    def insert_id(self, a):
        for i, v in enumerate(a):
            if v == self.src_dict.placeholder_index:
                return i
        raise Exception('Cannot find placeholder.')

    def __getitem__(self, index):
        item = {
            'id': index,
            'source-template': self.src[index]['template'],
            'target': self.tgt[index] if self.tgt is not None else None,
        }
        if self.insert == 'deleted':
            # Ignore EOS
            item['source-insert'] = self.src[index]['deleted'][:-1]
        if self.combine == 'token' and self.insert != 'none':
            id_ = self.insert_id(item['source-template'])
            #template = torch.cat((item['source-template'], item['source-insert'], torch.LongTensor([self.src_dict.eos()])), dim=0)
            template = torch.cat((
                item['source-template'][:id_],
                torch.LongTensor([self.src_dict.soi_index]),
                item['source-insert'],
                torch.LongTensor([self.src_dict.eoi_index]),
                item['source-template'][id_+1:],
                ), dim=0)
            item['source-template'] = template
            #print([self.src_dict[x] for x in item['source-template']])
            #print([self.tgt_dict[x] for x in item['target']])
            #import sys; sys.exit()
        return item

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        insert = 'none' if self.combine == 'token' else self.insert
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            insert=insert,
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        src_len, tgt_len = utils.resolve_max_positions(
            (src_len, tgt_len),
            max_positions,
            (self.max_source_positions, self.max_target_positions),
        )
        bsz = num_tokens // max(src_len, tgt_len)
        return self.collater([
            {
                'id': i,
                'source-template': self.src_dict.dummy_sentence(src_len),
                'source-insert': self.src_dict.dummy_sentence(1),
                'target': self.tgt_dict.dummy_sentence(tgt_len) if self.tgt_dict is not None else None,
            }
            for i in range(bsz)
        ])



# test
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    args = parser.parse_args()

    src_data = '{}/train.src-tgt.src'.format(args.data)
    src_dict = '{}/dict.src.txt'.format(args.data)
    tgt_data = '{}/train.src-tgt.tgt'.format(args.data)
    tgt_dict = '{}/dict.tgt.txt'.format(args.data)

    def indexed_dataset(path, dictionary):
        if IndexedInMemoryDataset.exists(path):
            return IndexedInMemoryDataset(path, fix_lua_indexing=True)
        return None

    src_dataset = indexed_dataset(src_data, src_dict)
    wrapped_src_dataset = SrcDatasetWrapper(indexed_dataset(src_data, src_dict))
    tgt_dataset = indexed_dataset(tgt_data, tgt_dict)
    print(src_dataset.size)
    print(src_dataset.sizes)
    print(src_dataset.sizes.dtype)
    print(wrapped_src_dataset.size)
    print(wrapped_src_dataset.sizes)
    print(wrapped_src_dataset.sizes.dtype)
    print(tgt_dataset.size)
    print(tgt_dataset.sizes)
