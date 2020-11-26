from flask import Flask, render_template, url_for, request
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
app = Flask(__name__)


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


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/generate', methods=['POST'])
def generate():
    args = parse_args()
    ensure_exist(args.outdir, is_dir=True)
    # json.dump(vars(args), open(os.path.join(args.outdir, 'config.json'), 'w'))

    # unigram_model = UnigramModel(args.word_counts_path, args.oov_prob)
    retriever = Retriever(args.doc_file, path=args.retriever_model, overwrite=args.overwrite_retriever_model)

    if args.system.startswith('rule') or args.system == 'keywords' or args.scorer in ('goodman',):
        skipgram = SkipGram.load_model(args.skipgram_model[0], args.skipgram_model[1],
                                       embedding_size=args.skipgram_embed_size, cpu=args.cpu)
    else:
        skipgram = None

    scorer = RandomScorer()
    type_recognizer = TypeRecognizer(threshold=args.type_consistency_threshold)
    if args.system == 'rule':
        generator = RulebasedGenerator(retriever, skipgram, type_recognizer, scorer,
                                   dist_to_pun=args.distance_to_pun_word)
    elif args.system == 'retrieve':
        generator = RetrieveGenerator(retriever, scorer)
    elif args.system == 'retrieve+swap':
        generator = RetrieveSwapGenerator(retriever, scorer)

    if request.method == 'POST':
        message = request.form['message']
        alter_word, pun_word = message.split()
        results = generator.generate(alter_word, pun_word, k=args.num_topic_words, ncands=args.num_candidates,
                                     ntemps=args.num_templates)
        results = [r for r in results if r.get('score') is not None]
        if len(results) > 0:
            results = sorted(results, key=lambda r: r['score'], reverse=True)
            response = '{}'.format(''.join(results[0]['output']))
        else:
            response = "你输入的可替代词与双关词无法生成合适的双关句"
    return render_template('result.html', response=response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7776, debug=True)
