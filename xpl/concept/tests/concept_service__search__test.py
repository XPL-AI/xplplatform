import pytest
from typing import List

from google.cloud import firestore

from xpl.concept import concept_service
from xpl.concept.concept_service import Concept

concept_service.CONCEPT_COLLECTION_NAME = 'concepts_autotest'
FIRESTORE_CLIENT = firestore.Client()


def __cleanup():
    global FIRESTORE_CLIENT
    docs = FIRESTORE_CLIENT.collection('concepts_autotest').stream()
    for doc in docs:
        doc.reference.delete()


@pytest.fixture
def registered_concepts():
    __cleanup()
    concepts_to_create = [
        Concept(concept_id='xpl:wn:n13742358', name='zero',
                definition='a mathematical element that when added to another number yields the same number',
                wn_id='wn:n13742358', bn_id=None, pos='NOUN'),
        Concept(concept_id='xpl:wn:n13742573', name='one', definition='the smallest whole number or a numeral representing this number',
                wn_id='wn:n13742573', pos='NOUN'),
        Concept(concept_id='xpl:wn:n13743269', name='two',
                definition='the cardinal number that is the sum of one and one or a numeral representing this number',
                wn_id='wn:n13743269', pos='NOUN'),
        Concept(concept_id='xpl:wn:n13744044', name='three', definition='the cardinal number that is the sum of one and one and one',
                wn_id='wn:n13744044', pos='NOUN'),

        Concept(concept_id='xpl:wn:n3182795', name='two', definition='one of the four playing cards in a deck that have two spots',
                wn_id='wn:n3182795', pos='NOUN'),

        Concept(concept_id='xpl:wn:n7846', name='person', definition='a human being', wn_id='wn:n7846', pos='NOUN'),
        Concept(concept_id='xpl:wn:n2084071', name='dog', definition='a member of the genus Canis (probably descended from the common wolf)'
                                                                 'that has been domesticated by man since prehistoric times; '
                                                                 'occurs in many breeds',
                wn_id='wn:n2084071', pos='NOUN'),
        Concept(concept_id='xpl:own_identifier_of_stargate', name='stargate', definition='a device that teleports to another galaxy.',
                pos='NOUN'),

        Concept(concept_id='xpl:wn:made_up_non_existing_id', name='not in wordnet', definition='does not exists in wordnet',
                wn_id='wn:made_up_non_existing_id', pos='NOUN')
    ]

    concepts = {}
    for concept in concepts_to_create:
        concept_service.create(concept=concept)
        concepts[concept.concept_id] = concept

    return concepts


def test__search__given_lemma__exists_in_xpl_and_wordnet__union_of_candidates_returned(registered_concepts: dict[Concept]):
    found_concepts = concept_service.search('two')
    assert len(found_concepts) > 0

    concepts_exist_in_both = []
    for concept in found_concepts:
        if f'xpl:{concept.wn_id}' in registered_concepts:
            concepts_exist_in_both.append(concept)

    present_in_wordnet = __to_wordnet_id_dict(found_concepts)
    present_in_xpl = __to_concept_id_dict(found_concepts)

    assert len(present_in_xpl) == 2
    xpl_two_wn_n13743269 = present_in_xpl['xpl:wn:n13743269']
    assert xpl_two_wn_n13743269 is not None
    assert xpl_two_wn_n13743269.concept_id == 'xpl:wn:n13743269'
    assert xpl_two_wn_n13743269.name == 'two'
    assert xpl_two_wn_n13743269.definition == 'the cardinal number that is the sum of one and one or a numeral representing this number'
    assert xpl_two_wn_n13743269.bn_id is None
    assert xpl_two_wn_n13743269.wn_id == 'wn:n13743269'
    assert xpl_two_wn_n13743269.pos == 'NOUN'
    assert 'wn:n13743269' in present_in_wordnet

    xpl_two_wn_n3182795 = present_in_xpl['xpl:wn:n3182795']
    assert xpl_two_wn_n3182795 is not None
    assert xpl_two_wn_n3182795.concept_id == 'xpl:wn:n3182795'
    assert xpl_two_wn_n3182795.name == 'two'
    assert xpl_two_wn_n3182795.definition == 'one of the four playing cards in a deck that have two spots'
    assert xpl_two_wn_n3182795.bn_id is None
    assert xpl_two_wn_n3182795.wn_id == 'wn:n3182795'
    assert xpl_two_wn_n3182795.pos == 'NOUN'
    assert 'wn:n3182795' in present_in_wordnet

    wordnet_two_wn_s2186470 = present_in_wordnet['wn:s2186470']
    assert wordnet_two_wn_s2186470 is not None
    assert wordnet_two_wn_s2186470.concept_id is None
    assert wordnet_two_wn_s2186470.name == 'two'
    assert wordnet_two_wn_s2186470.definition == 'being one more than one'
    assert wordnet_two_wn_s2186470.bn_id is None
    assert wordnet_two_wn_s2186470.wn_id == 'wn:s2186470'
    assert wordnet_two_wn_s2186470.pos == 'ADJECTIVE SATELLITE'
    assert 'xpl:wn:s2186470' not in present_in_xpl


def test__search__given_lemma__exists_in_xpl_but_absent_in_wordnet__xpl_candidates_returned(registered_concepts):
    found_concepts = concept_service.search('stargate')

    present_in_wordnet = __to_wordnet_id_dict(found_concepts)
    assert len(present_in_wordnet) == 0

    present_in_xpl = __to_concept_id_dict(found_concepts)
    assert len(present_in_xpl) == 1
    xpl_stargate_own_identifier = present_in_xpl['xpl:own_identifier_of_stargate']
    assert xpl_stargate_own_identifier is not None
    assert xpl_stargate_own_identifier.concept_id == 'xpl:own_identifier_of_stargate'
    assert xpl_stargate_own_identifier.name == 'stargate'
    assert xpl_stargate_own_identifier.definition == 'a device that teleports to another galaxy.'
    assert xpl_stargate_own_identifier.bn_id is None
    assert xpl_stargate_own_identifier.wn_id is None


def test__search__given_lemma__absent_in_xpl_but_present_in_wordnet__wordnet_candidates_returned(registered_concepts):
    """probably most common scenario in the beginning"""
    found_concepts = concept_service.search('cat')

    present_in_xpl = __to_concept_id_dict(found_concepts)
    assert len(present_in_xpl) == 0

    present_in_wordnet = __to_wordnet_id_dict(found_concepts)
    assert len(present_in_wordnet) > 0

    wordnet_cat_wn_n2121620 = present_in_wordnet['wn:n2121620']
    assert wordnet_cat_wn_n2121620 is not None
    assert wordnet_cat_wn_n2121620.concept_id is None
    assert wordnet_cat_wn_n2121620.name == 'cat'
    assert wordnet_cat_wn_n2121620.definition == 'feline mammal usually having thick soft fur and ' \
                                                 'no ability to roar: domestic cats; wildcats'
    assert wordnet_cat_wn_n2121620.bn_id is None
    assert wordnet_cat_wn_n2121620.wn_id == 'wn:n2121620'
    assert wordnet_cat_wn_n2121620.pos == 'NOUN'


def test__search__given_lemma__absent_in_xpl_absent_in_wordnet__empty_list_returned(registered_concepts):
    found_concepts = concept_service.search('nonexistingcompbination')

    present_in_xpl = __to_concept_id_dict(found_concepts)
    assert len(present_in_xpl) == 0

    present_in_wordnet = __to_wordnet_id_dict(found_concepts)
    assert len(present_in_wordnet) == 0


def test__search__given_wordnet_id__exists_in_xpl_and_wordnet__xpl_variant_returned(registered_concepts):
    wordnet_id = 'wn:n13743269'  # a wordnet's id for the concept of number 2
    found_concepts = concept_service.search(wordnet_id)
    assert len(found_concepts) == 1

    concept = found_concepts[0]
    assert concept is not None
    assert concept.concept_id == 'xpl:wn:n13743269'
    assert concept.name == 'two'
    assert concept.definition == 'the cardinal number that is the sum of one and one or a numeral representing this number'
    assert concept.bn_id is None
    assert concept.wn_id == 'wn:n13743269'


def test__search__given_wordnet_id__exists_in_xpl_but_absent_in_wordnet__xpl_variant_returned__error_sent_to_log(registered_concepts):
    """ this is strange condition. how did we appear here?
        we hold the reference to wordnet id that does not exist.
        return xpl's concept and log error.
    """
    wordnet_id = 'wn:made_up_non_existing_id'
    found_concepts = concept_service.search(wordnet_id)
    assert len(found_concepts) == 1

    concept = found_concepts[0]
    assert concept is not None
    assert concept.concept_id == 'xpl:wn:made_up_non_existing_id'
    assert concept.name == 'not in wordnet'
    assert concept.definition == 'does not exists in wordnet'
    assert concept.bn_id is None
    assert concept.wn_id == 'wn:made_up_non_existing_id'


def test__search__given_wordnet_id__absent_in_xpl_but_present_in_wordnet__wordnet_variant_returned(registered_concepts):
    wordnet_id = 'wn:n2121620'  # a wordnet's id for the concept of a cat
    found_concepts = concept_service.search(wordnet_id)
    assert len(found_concepts) == 1

    concept = found_concepts[0]
    assert concept is not None
    assert concept.concept_id is None
    assert concept.name == 'cat'
    assert concept.definition == 'feline mammal usually having thick soft fur and no ability to roar: domestic cats; wildcats'
    assert concept.bn_id is None
    assert concept.wn_id == 'wn:n2121620'


def test__search__given_wordnet_id__absent_in_xpl_absent_in_wordnet__empty_list_returned(registered_concepts):
    wordnet_id = 'wn:n5656565656565'
    found_concepts = concept_service.search(wordnet_id)
    assert len(found_concepts) == 0


def test__search__given_xpl_id__present__single_xpl_concept_returned(registered_concepts):
    xpl_concept_id = 'xpl:wn:n7846'  # an XPL's id for human being
    found_concepts = concept_service.search(xpl_concept_id)
    assert len(found_concepts) == 1

    concept = found_concepts[0]
    assert concept is not None
    assert concept.concept_id == xpl_concept_id
    assert concept.name == 'person'
    assert concept.definition == 'a human being'
    assert concept.bn_id is None
    assert concept.wn_id == 'wn:n7846'


def test__search__given_xpl_id__absent__empty_list_returned(registered_concepts):
    xpl_concept_id = 'xpl:wn:not_exist'  # an XPL's id for human being
    found_concepts = concept_service.search(xpl_concept_id)
    assert len(found_concepts) == 0


def __to_wordnet_id_dict(concepts: List[Concept]):
    dictionary = {}
    for concept in concepts:
        if concept.wn_id is not None:
            dictionary[concept.wn_id] = concept
    return dictionary


def __to_concept_id_dict(concepts: List[Concept]):
    dictionary = {}
    for concept in concepts:
        if concept.concept_id is not None:
            dictionary[concept.concept_id] = concept
    return dictionary
