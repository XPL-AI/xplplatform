import importlib
import os
import pytest
from typing import List

from google.cloud import firestore

from xpl.concept import config
from xpl.concept import concept_service
from xpl.concept.concept_service import Concept
from xpl.concept import concept_resolver
from xpl.concept.concept_resolver import NoAppropriateConceptException


concept_service.CONCEPT_COLLECTION_NAME = 'concepts_autotest'
FIRESTORE_CLIENT = firestore.Client()


def __cleanup(local_concept_id_name):
    mapping_file_path = os.path.join(config.DATA_DIR, 'concept', f'{local_concept_id_name}.csv')
    if os.path.exists(mapping_file_path):
        os.remove(mapping_file_path)

    global FIRESTORE_CLIENT
    docs = FIRESTORE_CLIENT.collection('concepts_autotest').stream()
    for doc in docs:
        doc.reference.delete()


def test__local_mapping__accuracy_scenario():
    local_concept_collection = 'autotest_mnist'

    __cleanup(local_concept_collection)

    # 1. Try to resolve mapping of local concept id.
    local_concept_name = 'zero'
    concept_id = concept_resolver.get_concept_id_from_local_concept_name(local_concept_collection, local_concept_name)

    # Expected: mapping does not exist
    assert concept_id is None

    # 2. Search for the candidates by lemma.
    candidates = concept_service.search(local_concept_name)
    # Expected: there exist candidates
    assert len(candidates) > 0

    # 3. Choose concept
    chosen_concept = choose(candidates, 1)
    assert chosen_concept.definition == 'a mathematical element that when added to another number yields the same number'
    assert chosen_concept.name == 'zero'
    assert chosen_concept.concept_id is None
    assert chosen_concept.wn_id == 'wn:n13742358'

    # 4. Add mapping for the chosen concept
    concept_id = concept_resolver.add_local_mapping(local_concept_collection, local_concept_name, chosen_concept)
    assert concept_id == 'xpl:wn:n13742358'

    # ######## Verifications ########

    # 5. Try to resolve mapping of local concept id (1.) once again.
    concept_id = concept_resolver.get_concept_id_from_local_concept_name(local_concept_collection, local_concept_name)

    # Expected:
    # Mapping exists; concept is in database
    assert concept_id is not None
    assert concept_id == 'xpl:wn:n13742358'

    concept = concept_service.get_by_id(concept_id=concept_id)
    assert concept is not None
    assert concept.concept_id == 'xpl:wn:n13742358'
    assert concept.name == 'zero'
    assert concept.definition == 'a mathematical element that when added to another number yields the same number'
    assert concept.bn_id is None
    assert concept.wn_id == 'wn:n13742358'

    # Reload concept_service module to reload concept mapping cache. Then repeat verifications
    # importlib.reload(concept_service)
    # assert concept_service.CONCEPT_COLLECTION_NAME == 'concepts'  # confirmation of reload
    # concept_service.CONCEPT_COLLECTION_NAME = 'concepts_autotest'

    concept_id = concept_resolver.get_concept_id_from_local_concept_name(local_concept_collection, local_concept_name)

    # Expected:
    # Mapping exists; concept is in database
    assert concept_id is not None
    assert concept_id == 'xpl:wn:n13742358'
    concept = concept_service.get_by_id(concept_id=concept_id)
    assert concept is not None
    assert concept.concept_id == 'xpl:wn:n13742358'
    assert concept.name == 'zero'
    assert concept.definition == 'a mathematical element that when added to another number yields the same number'
    assert concept.bn_id is None
    assert concept.wn_id == 'wn:n13742358'


def test__local_mapping__xpl_concept_exist__no_mapping__mapping_added():
    local_concept_collection = 'autotest_mnist'
    __cleanup(local_concept_collection)

    concept_service.create(Concept(
        concept_id='xpl:wn:n13742573',
        wn_id='wn:n13742573',
        name='one',
        definition='the smallest whole number or a numeral representing this number'))

    local_concept_name = 'one'
    concept_id = concept_resolver.get_concept_id_from_local_concept_name(local_concept_collection, local_concept_name)

    assert concept_id is None
    candidates = concept_service.search(local_concept_name)
    assert len(candidates) > 0

    chosen_concept = choose(candidates, 0)
    assert chosen_concept.concept_id == 'xpl:wn:n13742573'
    assert chosen_concept.wn_id == 'wn:n13742573'

    concept_id = concept_resolver.add_local_mapping(local_concept_collection, local_concept_name, chosen_concept)

    assert concept_id == 'xpl:wn:n13742573'


def test__local_mapping__xpl_concept_not_exist__no_mapping__concept_created_mapping_added():
    local_concept_collection = 'autotest_mnist'
    __cleanup(local_concept_collection)

    local_concept_name = 'two'
    concept_id = concept_resolver.get_concept_id_from_local_concept_name(local_concept_collection, local_concept_name)

    assert concept_id is None
    candidates = concept_service.search(local_concept_name)
    assert len(candidates) > 0

    chosen_concept = choose(candidates, 0)
    assert chosen_concept.concept_id is None
    assert chosen_concept.wn_id == 'wn:n13743269'
    assert chosen_concept.name == 'two'
    assert chosen_concept.definition == 'the cardinal number that is the sum of one and one or a numeral representing this number'

    anticipated_concept_id = f'xpl:{chosen_concept.wn_id}'
    assert concept_service.get_by_id(anticipated_concept_id) is None

    concept_id = concept_resolver.add_local_mapping(local_concept_collection, local_concept_name, chosen_concept)

    assert concept_id == anticipated_concept_id
    new_concept = concept_service.get_by_id(concept_id)
    assert new_concept is not None
    assert new_concept.name == chosen_concept.name
    assert new_concept.definition == chosen_concept.definition


def test__local_mapping__nothing_in_system__no_options_in_wordnet___created_concept_from_name_mapping_added():
    local_concept_collection = 'autotest_mnist'
    __cleanup(local_concept_collection)

    local_concept_name = 'antigravity'
    concept_id = concept_resolver.get_concept_id_from_local_concept_name(local_concept_collection, local_concept_name)

    assert concept_id is None
    candidates = concept_service.search(local_concept_name)
    assert len(candidates) == 0

    new_concept_id = None
    try:
        choose(candidates, -1)
    except NoAppropriateConceptException:
        new_concept_id = concept_resolver.add_local_mapping(local_concept_collection, local_concept_name, Concept(name=local_concept_name))

    new_concept = concept_service.get_by_id(new_concept_id)
    assert new_concept is not None
    assert new_concept.name == local_concept_name
    assert new_concept.definition is None
    assert new_concept.wn_id is None
    assert new_concept.bn_id is None


def choose(concepts: List[Concept], choice: int):
    if choice >= 0:
        return concepts[choice]
    else:
        raise NoAppropriateConceptException()
