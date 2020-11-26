import os
import pickle
import argparse
import json
from collections import defaultdict
import fuzzy

from fairseq import options

from pungen.retriever import Retriever
from pungen.generator import SkipGram, RulebasedGenerator, NeuralCombinerGenerator, RetrieveGenerator, RetrieveSwapGenerator, KeywordsGenerator
from pungen.scorer import LMScorer, SurprisalScorer, UnigramModel, RandomScorer, GoodmanScorer
from pungen.type import TypeRecognizer
from pungen.options import add_scorer_args, add_editor_args, add_retriever_args, add_generic_args, add_type_checker_args
from pungen.utils import logging_config, get_lemma, ensure_exist, get_spacy_nlp

import logging
logger = logging.getLogger('pungen')

nlp = get_spacy_nlp()


def parse_args():
    parser = options.get_generation_parser(interactive=True)
    add_scorer_args(parser)
    add_editor_args(parser)
    add_retriever_args(parser)
    add_type_checker_args(parser)
    add_generic_args(parser)
    parser.add_argument('--pun-words')
    parser.add_argument('--system', default='rule')
    parser.add_argument('--max-num-examples', type=int, default=-1)
    args = options.parse_args_and_arch(parser)
    return args


def iter_keywords(file_):
    with open(file_, 'r') as fin:
        for line in fin:
            alter_word, pun_word = line.strip().split()
            yield alter_word, pun_word


def feasible_pun_words(pun_word, alter_word, skipgram=None, freq_threshold=1000):
    # Pun / alternative word cannot be phrases
    # if len(alter_word.split('_')) > 1 or len(pun_word.split('_')) > 1:
    #     logger.info('FAIL: phrase')
    #     return False, 'phrase'

    if skipgram and skipgram.vocab.index(get_lemma(pun_word)) == skipgram.vocab.unk():
        logger.info('FAIL: unknown pun word: {}'.format(pun_word))
        return False, 'unk to skipgram'

    return True, None


def main(args):
    ensure_exist(args.outdir, is_dir=True)
    # json.dump(vars(args), open(os.path.join(args.outdir, 'config.json'), 'w'))

    # unigram_model = UnigramModel(args.word_counts_path, args.oov_prob)
    retriever = Retriever(args.doc_file, path=args.retriever_model, overwrite=args.overwrite_retriever_model)

    if args.system.startswith('rule') or args.system == 'keywords' or args.scorer in ('goodman',):
        skipgram = SkipGram.load_model(args.skipgram_model[0], args.skipgram_model[1],
                                       embedding_size=args.skipgram_embed_size, cpu=args.cpu)
    else:
        skipgram = None
    # if args.scorer == 'random':
    #     scorer = RandomScorer()
    # elif args.scorer == 'surprisal':
    #     lm = LMScorer.load_model(args.lm_path)
    #     scorer = SurprisalScorer(lm, unigram_model, local_window_size=args.local_window_size)
    # elif args.scorer == 'goodman':
    #     scorer = GoodmanScorer(unigram_model, skipgram)
    scorer = RandomScorer()
    type_recognizer = TypeRecognizer(threshold=args.type_consistency_threshold)

    if args.system == 'rule':
        generator = RulebasedGenerator(retriever, skipgram, type_recognizer, scorer,
                                       dist_to_pun=args.distance_to_pun_word)
    elif args.system == 'rule+neural':
        generator = NeuralCombinerGenerator(retriever, skipgram, type_recognizer, scorer, args.distance_to_pun_word,
                                            args)
    elif args.system == 'retrieve':
        generator = RetrieveGenerator(retriever, scorer)
    elif args.system == 'retrieve+swap':
        generator = RetrieveSwapGenerator(retriever, scorer)

    if args.interactive:
        alter_word, pun_word = input('Keywords:\n').split()
        # logger.info('-' * 50)
        print('-' * 50)
        # logger.info('INPUT: alter={} pun={}'.format(alter_word, pun_word))
        print('INPUT: alter={} pun={}'.format(alter_word, pun_word))
        # logger.info('-' * 50)
        print('-' * 50)
        # feasible, reason = feasible_pun_words(pun_word, alter_word, skipgram=skipgram,
        #                                       freq_threshold=args.pun_freq_threshold)
        results = generator.generate(alter_word, pun_word, k=args.num_topic_words, ncands=args.num_candidates,
                                   ntemps=args.num_templates)
        results = [r for r in results if r.get('score') is not None]
        if len(results) > 0:
            results = sorted(results, key=lambda r: r['score'], reverse=True)
            for r in results[:5]:
                # logger.info('{:<8.2f}{}'.format(r['score'], ' '.join(r['output'])))
                print('{}'.format(''.join(r['output'])))
        else:
            print("你输入的可替代词与双关词无法生成合适的双关句，可尝试修改生成模式为 --system retrieve+swap ")


if __name__ == '__main__':
    args = parse_args()
    main(args)




