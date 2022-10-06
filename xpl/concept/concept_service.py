from typing import Optional
from pydantic import BaseModel

from xpl.concept.repositories import firestore
from xpl.concept import config
from xpl.concept import wordnet_service


class Concept(BaseModel):
    concept_id: Optional[str]
    name: str
    definition: Optional[str]
    bn_id: Optional[str]
    wn_id: Optional[str]
    pos: Optional[str]


PART_OF_SPEECH_WORDNET_DICT = {
    'n': 'NOUN',
    # '': 'PRONOUN',
    'v': 'VERB',
    'a': 'ADJECTIVE',
    's': 'ADJECTIVE SATELLITE',
    'r': 'ADVERB',
    # '': 'PREPOSITION',
    # '': 'CONJUNCTION',
    # '': 'INTERJECTION'
}
""" https://wordnet.princeton.edu/frequently-asked-questions
WordNet only contains "open-class words": nouns, verbs, adjectives, and adverbs.
Thus, excluded words include determiners, prepositions, pronouns, conjunctions, and particles.
"""


# TODO: this will throw Nulleref if value is not in config. Find a way with configs.
CONCEPT_COLLECTION_NAME = config['concept_collection_name']
CONCEPT_COLLECTION_NAME = 'concepts' if CONCEPT_COLLECTION_NAME is None else CONCEPT_COLLECTION_NAME
CONCEPT_SEARCH_PROVIDER = config['concept_search_provider']
CONCEPT_SEARCH_PROVIDER = 'wordnet' if CONCEPT_SEARCH_PROVIDER is None else CONCEPT_SEARCH_PROVIDER


def create(concept: Concept):
    if concept.concept_id is None:
        raise CreateConceptException(f'Concept.concept_id should be provided by the caller.')

    existing = get_by_id(concept.concept_id)
    if existing is not None:
        raise CreateConceptException(f'Concept.concept_id={concept.concept_id} already exists in collection={CONCEPT_COLLECTION_NAME}')

    if concept.bn_id is not None:
        existing = get_by_babelnet_id(bn_id=concept.bn_id)
        if existing is not None:
            raise CreateConceptException(f'Concept.bn_id={concept.bn_id} already exists in collection={CONCEPT_COLLECTION_NAME}.')

    if concept.wn_id is not None:
        existing = get_by_wordnet_id(wordnet_id=concept.wn_id)
        if existing is not None:
            raise CreateConceptException(f'Concept.wn_id={concept.wn_id} already exists in collection={CONCEPT_COLLECTION_NAME}.')

    firestore.create_document(document_id=concept.concept_id, document=concept.dict(), collection=CONCEPT_COLLECTION_NAME)


def update(concept_id: str, name: str = None, definition: str = None, wn_id: str = None, bn_id: str = None):
    if concept_id is None:
        raise CreateConceptException(f'concept_id should be provided by the caller.')

    concept_to_update = get_by_id(concept_id)
    if concept_to_update is None:
        raise UpdateConceptException(f'concept_id={concept_id} does not exists.')

    if name is not None:
        concept_to_update.name = name
    if definition is not None:
        concept_to_update.definition = definition
    if wn_id is not None:
        concept_to_update.wn_id = wn_id
        existing = get_by_wordnet_id(wordnet_id=wn_id)
        if existing is not None:
            raise UpdateConceptAlreadyExistsException(f'Concept.bn_id={bn_id} already exists.')
    if bn_id is not None:
        existing = get_by_babelnet_id(bn_id=bn_id)
        if existing is not None:
            raise UpdateConceptAlreadyExistsException(f'Concept.bn_id={bn_id} already exists.')
        concept_to_update.bn_id = bn_id

    firestore.upsert_document(document_id=concept_id, document=concept_to_update.dict(), collection=CONCEPT_COLLECTION_NAME)


def get_by_id(concept_id):
    """
    When Concept Id in XPL system is known, this method should be used to fetch it.
    @param concept_id: An XPL's internal concept identifier.
    @return:
    """
    concept_doc = firestore.get_document_by_id(document_id=concept_id, collection=CONCEPT_COLLECTION_NAME)

    if concept_doc is None:
        return None

    return Concept(**concept_doc)


def get_by_external_id(external_id: str):
    """
    Resolves any valid and known to XPL identifier to particular identifier
    then: looks for the concept using particular identifier.
    @param external_id: an identifier in universal format
            identity_provider:id
            Examples:
    wordnet's wn_id     wn:n013745270
    babelnet's id           bn:00000005n
    google open image id    openimage:/m/03bt1vf
    COCO label              coco:/m/03bt1vf
    @return: A Concept if found
    """
    if external_id is None:
        raise Exception('Invalid input')

    external_id = external_id.strip()
    if not external_id:
        raise Exception('Invalid input')

    concept = None

    prefix = external_id.split(':')[0].strip()

    if prefix == 'wn':
        concept = get_by_wordnet_id(external_id)
        if concept is None:
            synset = wordnet_service.get_synset(external_id)
            if synset is not None:
                concept = Concept(
                    name=synset.lemma_names()[0],
                    definition=synset.definition(),
                    wn_id=f'wn:{synset.pos()}{synset.offset()}'
                )

    if prefix == 'bn':
        concept = get_by_babelnet_id(external_id)

    return concept


def search(user_input_text: str):
    """Searches for concepts by user-typed text"""
    if user_input_text is None:
        raise Exception('Invalid input')

    text = user_input_text.strip().lower()
    if not text:
        raise Exception('Invalid input')

    # first try to fetch concept assuming we received XPL's identifier
    # XPL identifier examples:
    #   xpl:user:7e0ccaef027e
    #   xpl:wn:n13742573
    #   xpl:bn:00000005n
    prefix = text.split(':')[0].strip()
    if prefix == 'xpl':
        concept = get_by_id(text)
        if concept is not None:
            return [concept]
        else:
            return []

    # try to find concept assuming we received one of external identifiers:
    try:
        concept = get_by_external_id(text)
        if concept is not None:
            return [concept]
    except Exception:
        pass

    registered_with_matching_name = firestore.get_documents_by_field(field_name='name',
                                                                     field_value=text,
                                                                     collection=CONCEPT_COLLECTION_NAME)

    result = []
    # depending on which concept provider we rely - we will search it for concepts matching the lemma=user_input_text
    if CONCEPT_SEARCH_PROVIDER == 'wordnet':
        synsets = wordnet_service.search(text)
        for s in synsets:
            wordnet_id = f'wn:{s.pos()}{s.offset()}'
            concept = get_by_wordnet_id(wordnet_id)
            if concept is None:
                concept = Concept(
                    name=s.lemma_names()[0],
                    definition=s.definition(),
                    wn_id=wordnet_id,
                    pos=PART_OF_SPEECH_WORDNET_DICT[s.pos()]
                )
            result.append(concept)
        for registered in registered_with_matching_name:
            if 'wn_id' not in registered or registered['wn_id'] is None:
                result.append(Concept(**registered))

    elif CONCEPT_SEARCH_PROVIDER == 'babelnet':
        raise NotImplemented('Concept unit is not yet integrated with babelnet')

    return result


def get_by_wordnet_id(wordnet_id):
    return __get_by_external_id(external_id_name='wn_id', external_id=wordnet_id)


def get_by_babelnet_id(bn_id):
    return __get_by_external_id(external_id_name='bn_id', external_id=bn_id)


def __get_by_external_id(external_id_name, external_id):
    concept_documents = firestore.get_documents_by_field(field_name=external_id_name, field_value=external_id,
                                                         collection=CONCEPT_COLLECTION_NAME)
    count = len(concept_documents)

    if count == 0:
        return None

    if count == 1:
        concept = Concept(**concept_documents[0])
        return concept

    # Commented since it's not decided how to handle those. TODO: design duplicate elimination routine
    # raise Exception(f'There are duplicate values for same COCO label id coco_id="{coco_id}" in concepts collections')
    print(f'WARN There are duplicate values for same {external_id_name}="{external_id}" in concepts collections')
    return None


def load_dictionary_by_concept_identifier(identifier_name):
    """
    Loads all Concepts for which identifier_name is set.
    This method is designed to initialize cache.
    @return:
    """
    doc_dictionaries = firestore.get_documents_that_have_field(field_name=identifier_name, collection=CONCEPT_COLLECTION_NAME)

    concept_dictionary = {}
    for d in doc_dictionaries:
        concept = Concept(**d)
        concept_dictionary[concept.concept_id] = concept


class CreateConceptException(Exception):
    pass


class UpdateConceptException(Exception):
    pass


class UpdateConceptAlreadyExistsException(Exception):
    pass
