from typing import List

from os.path import join

import nltk
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Synset, WordNetError

from xpl.concept import config


CONCEPT_CODE_DIR = join(config.XPL_CODE_DIR, 'xpl', 'concept')
try:
    nltk.data.find('corpora/wordnet.zip', paths=[CONCEPT_CODE_DIR])

except LookupError as error:
    nltk.download('wordnet', download_dir=CONCEPT_CODE_DIR)
finally:
    nltk.data.path = [CONCEPT_CODE_DIR]


def search(text: str) -> List[Synset]:
    found_synsets = wordnet.synsets(text)
    return found_synsets


def get_synset(synset_id: str) -> Synset:
    pos = synset_id[3]

    offset = __safe_to_int(synset_id[4:], int)
    if offset is None:
        raise WordnetSynsetNotFoundException(f'No WordNet synset found for synset_id={synset_id}')

    synset = get_synset_by_pos_and_offset(pos, offset)
    return synset


def get_synset_by_pos_and_offset(pos: str, offset: int) -> Synset:
    try:
        synset = wordnet.synset_from_pos_and_offset(pos=pos, offset=offset)
        return synset
    except WordNetError as e:
        message: str = e.args[0]
        if message.startswith('No WordNet synset found'):
            raise WordnetSynsetNotFoundException(f'No WordNet synset found for pos={pos} offset={offset}')


def __safe_to_int(val, to_type, default=None):
    try:
        return to_type(val)
    except (ValueError, TypeError):
        return default


class WordnetSynsetNotFoundException(Exception):
    pass
