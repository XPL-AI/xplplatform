import os
import random
import pytest
import time

from google.cloud import firestore

from xpl.concept import concept_service
from xpl.concept.concept_service import Concept, CreateConceptException, UpdateConceptAlreadyExistsException


concept_service.CONCEPT_COLLECTION_NAME = 'concepts_autotest'
FIRESTORE_CLIENT = firestore.Client()


def __cleanup():
    global FIRESTORE_CLIENT
    docs = FIRESTORE_CLIENT.collection('concepts_autotest').stream()
    for doc in docs:
        doc.reference.delete()


def test__create__lucky_path__accuracy():
    __cleanup()

    concept_id = f'xpl:concept{random.randrange(10000)}'
    concept_to_create = Concept(
        concept_id=concept_id,
        name='name__test__create__lucky_path__accuracy',
        definition='definition__test__create__lucky_path__accuracy',
        wn_id='wn:s1925000',
        pos='ADJECTIVE SATELLITE'
    )

    concept_service.create(concept=concept_to_create)

    concept = concept_service.get_by_id(concept_id=concept_id)
    assert concept is not None
    assert concept.concept_id == concept_id
    assert concept.name == concept_to_create.name
    assert concept.definition == concept_to_create.definition
    assert concept.bn_id is None
    assert concept.wn_id == concept_to_create.wn_id
    assert concept.pos == 'ADJECTIVE SATELLITE'

    concept = concept_service.get_by_wordnet_id(wordnet_id=concept_to_create.wn_id)
    assert concept is not None
    assert concept.concept_id == concept_id
    assert concept.name == concept_to_create.name
    assert concept.definition == concept_to_create.definition
    assert concept.bn_id is None
    assert concept.wn_id == concept_to_create.wn_id
    assert concept.pos == 'ADJECTIVE SATELLITE'

    # Expected: external id is resolver to correct field and concept is found.
    concept = concept_service.get_by_external_id(external_id=concept_to_create.wn_id)
    assert concept is not None
    assert concept.concept_id == concept_id
    assert concept.name == concept_to_create.name
    assert concept.definition == concept_to_create.definition
    assert concept.bn_id is None
    assert concept.wn_id == concept_to_create.wn_id
    assert concept.pos == 'ADJECTIVE SATELLITE'


def test__create__concept_id_already_exists__exception():
    __cleanup()

    concept_id = f'xpl:concept{random.randrange(10000)}'
    concept_to_create = Concept(
        concept_id=concept_id,
        name='name__test__create__concept_id_already_exists__exception',
        wn_id='wn:n13742573',
        bn_id='bn:00000005n'
    )
    concept_service.create(concept=concept_to_create)

    # Case: concept_id already exists
    concept_to_create = Concept(
        concept_id=concept_id,
        name='concept id already exists',
        wn_id='wn:n13742573',
        bn_id='bn:00000005n'
    )

    with pytest.raises(CreateConceptException):
        concept_service.create(concept=concept_to_create)

    # Case: wordnet's id already exists
    concept_to_create = Concept(
        concept_id=f'xpl:concept{random.randrange(10000)}',
        name='wordnet id already exists',
        wn_id='wn:n13742573'
    )

    with pytest.raises(CreateConceptException):
        concept_service.create(concept=concept_to_create)

    # Case: babelnet id already exists
    concept_to_create = Concept(
        concept_id=f'xpl:concept{random.randrange(10000)}',
        name='babelnet id already exists',
        bn_id='bn:00000005n'
    )

    with pytest.raises(CreateConceptException):
        concept_service.create(concept=concept_to_create)


def test__create__concept_id_not_provided__exception():
    __cleanup()
    concept_to_create = Concept(
        # concept_id=concept_id,
        name='name__test__create__concept_id_already_exists__exception',
        wn_id='wn:n13742573',
        bn_id='bn:00000005n'
    )

    try:
        concept_service.create(concept=concept_to_create)
    except CreateConceptException as exc:
        assert exc.args[0] == 'Concept.concept_id should be provided by the caller.'


def test__update__accuracy():
    __cleanup()

    concept_id = f'xpl:concept{random.randrange(10000)}'
    concept_to_create = Concept(
        concept_id=concept_id,
        name='name__test__update__accuracy',
        definition='definition__test__update__accuracy',
        wn_id='wn:s1925000'
    )

    concept_service.create(concept_to_create)

    # Case: update definition
    new_definition = 'new definition'
    concept_service.update(concept_id=concept_id, definition=new_definition)
    # Expected: only definition is updated
    concept = concept_service.get_by_id(concept_id)
    assert concept.definition == new_definition

    assert concept.concept_id == concept_id
    assert concept.name == concept_to_create.name
    assert concept.bn_id is None
    assert concept.wn_id == concept_to_create.wn_id

    # Case: update name, bn_id and wn_id
    new_name = 'new name'
    new_bn_id = 'bn:00109425a'
    new_wn_id = 'wn:a1924316'
    concept_service.update(concept_id=concept_id, name=new_name, wn_id=new_wn_id, bn_id=new_bn_id)
    # Expected: corresponding fields updated
    concept = concept_service.get_by_id(concept_id)
    assert concept.definition == new_definition
    assert concept.concept_id == concept_id
    assert concept.name == new_name
    assert concept.bn_id == new_bn_id
    assert concept.wn_id == new_wn_id
    assert concept.pos is None


def test__update__already_exist__exception__accuracy():
    __cleanup()

    concept_id = f'xpl:concept{random.randrange(10000)}'
    concept_to_create = Concept(
        concept_id=concept_id,
        name='name__test__update__accuracy',
        definition='definition__test__update__accuracy',
        wn_id='wn:s1925000'
    )

    another_concept_to_create = Concept(
        concept_id=f'xpl:another_concept{random.randrange(10000)}',
        name='name__another_concept_to_create',
        definition='definition__another_concept_to_create',
        wn_id='wn:a1924316',
        bn_id='bn:00109425a'
    )

    concept_service.create(concept_to_create)
    concept_service.create(another_concept_to_create)

    # Case: try to update wn_id and bn_id to value that already exists.
    # Expected: Exception
    with pytest.raises(UpdateConceptAlreadyExistsException):
        concept_service.update(concept_id=concept_id, wn_id='wn:a1924316')
    with pytest.raises(UpdateConceptAlreadyExistsException):
        concept_service.update(concept_id=concept_id, bn_id='bn:00109425a')

    concept_service.update(concept_id=concept_id,
                           name=another_concept_to_create.name,
                           definition=another_concept_to_create.definition)

    concept = concept_service.get_by_id(concept_id)
    assert concept is not None
    assert concept.concept_id == concept_id

    assert concept.name == another_concept_to_create.name
    assert concept.definition == another_concept_to_create.definition

    assert concept.bn_id == concept_to_create.bn_id
    assert concept.wn_id == concept_to_create.wn_id
