import os
import inquirer
import pandas
from typing import List
import uuid

from xpl.concept import config

from xpl.concept import concept_service
from xpl.concept.concept_service import Concept


LOCAL_CONCEPT_MAPPINGS_CACHE = {}
"""
Holds cache for different local identity sources.
{
    'my_dataset': {'light': 'xpl:wn:n11473954', 'shadow': 'xpl:wn:n13984944'},
    'your_dataset': {'waves': 'xpl:wn:n7352190', 'clouds': 'xpl:wn:n11439690'},
}
"""


def resolve_to_concept_id(local_concept_collection: str, local_concept_name: str):
    if local_concept_name.startswith('xpl:'):
        return local_concept_name

    concept_id = get_concept_id_from_local_concept_name(local_concept_collection=local_concept_collection,
                                                        local_concept_name=local_concept_name)

    if concept_id is not None:
        return concept_id

    lemma = local_concept_name
    candidates = concept_service.search(lemma)

    try:
        selected = ask_user_to_choose(candidates, message=f'Please select the concept that describes "{lemma.upper()}" the best',
                                      lemma=lemma.upper())
    except NoAppropriateConceptException:
        selected = Concept(name=local_concept_name, definition=local_concept_name)

    concept_id = add_local_mapping(local_concept_collection=local_concept_collection, local_concept_name=local_concept_name, concept=selected)
    return concept_id


def ask_user_to_choose(concepts: List[Concept], message: str, lemma: str):
    choices = [f'0:  {"Not in the list.":24s}  Create new from the name={lemma}?']
    for idx, c in enumerate(concepts):
        if c.concept_id is not None:
            choice = f'{str(idx + 1) + ":":3s} {"(" + c.concept_id + ")":24s} ({c.pos})  {c.definition}'
        elif c.wn_id is not None:
            choice = f'{str(idx + 1) + ":":3s} {"(" + c.wn_id + ")":24s} ({c.pos}) {c.definition}'
        else:
            choice = f'{str(idx + 1) + ":":3s} {"(" "":24s + ")"} ({c.pos}) {c.name} {c.definition}'

        choices.append(choice)

    default_choice = 1 if len(concepts) > 0 else 0
    questions = [inquirer.List(name='_', message=message, choices=choices, default=choices[default_choice])]

    answers = inquirer.prompt(questions)

    answer = answers['_']
    idx = int(answer.split(':')[0].strip())

    if idx == 0:
        raise NoAppropriateConceptException()

    return concepts[idx - 1]


def get_concept_id_from_local_concept_name(local_concept_collection, local_concept_name: str):
    global LOCAL_CONCEPT_MAPPINGS_CACHE
    if local_concept_collection not in LOCAL_CONCEPT_MAPPINGS_CACHE:
        __reload_local_concept_mapping_cache(local_concept_collection)

    if local_concept_name in LOCAL_CONCEPT_MAPPINGS_CACHE[local_concept_collection]:
        return LOCAL_CONCEPT_MAPPINGS_CACHE[local_concept_collection][local_concept_name]

    return None


def add_local_mapping(local_concept_collection: str, local_concept_name: str, concept: Concept):
    global LOCAL_CONCEPT_MAPPINGS_CACHE
    if local_concept_collection not in LOCAL_CONCEPT_MAPPINGS_CACHE:
        __reload_local_concept_mapping_cache(local_concept_collection)

    if concept.concept_id is None:
        """Concept has not been introduced at XPL before - so we create it in the the system"""
        if concept_service.CONCEPT_SEARCH_PROVIDER == 'wordnet':
            if concept.wn_id is not None:
                concept.concept_id = f'xpl:{concept.wn_id}'
            else:
                concept.concept_id = f'xpl:user:{str(uuid.uuid4())[-12:]}'
            concept_service.create(concept)
        else:
            raise NotImplemented(f'CONCEPT_SEARCH_PROVIDER={concept_service.CONCEPT_SEARCH_PROVIDER} is not implemented')

    mappings = LOCAL_CONCEPT_MAPPINGS_CACHE[local_concept_collection]
    if local_concept_name not in mappings:
        mappings[local_concept_name] = concept.concept_id
    else:
        raise LocalMappingException(
            f'Mapping for {local_concept_collection}={local_concept_name} already exists: concept_id={mappings[local_concept_name]}')

    __save_local_concept_mapping_cache(local_concept_collection)

    return concept.concept_id


def __reload_local_concept_mapping_cache(local_concept_collection):
    global LOCAL_CONCEPT_MAPPINGS_CACHE
    concept_dir = os.path.join(config.DATA_DIR, 'concept')
    mapping_file_path = os.path.join(concept_dir, f'{local_concept_collection}.csv')
    if os.path.exists(mapping_file_path):
        data_frame = pandas.read_csv(mapping_file_path)
        LOCAL_CONCEPT_MAPPINGS_CACHE[local_concept_collection] = {data_frame['local_concept_name'][i]: data_frame['concept_id'][i]
                                                                  for i in range(len(data_frame))}
    else:
        LOCAL_CONCEPT_MAPPINGS_CACHE[local_concept_collection] = {}


def __save_local_concept_mapping_cache(local_concept_collection):
    global LOCAL_CONCEPT_MAPPINGS_CACHE
    concept_dir = os.path.join(config.DATA_DIR, 'concept')
    if not os.path.exists(concept_dir):
        os.makedirs(concept_dir)

    local_concept_ids = LOCAL_CONCEPT_MAPPINGS_CACHE[local_concept_collection].keys()
    concept_ids = LOCAL_CONCEPT_MAPPINGS_CACHE[local_concept_collection].values()

    mapping_file_path = os.path.join(concept_dir, f'{local_concept_collection}.csv')
    data_frame = pandas.DataFrame({'local_concept_name': local_concept_ids, 'concept_id': concept_ids})
    data_frame.to_csv(mapping_file_path, index=False)


class LocalMappingException(Exception):
    pass


class NoAppropriateConceptException(Exception):
    pass


if __name__ == "__main__":
    while True:
        local_mapping_name = 'local_concept_collection_playground'
        concept_name = input("Enter local_concept_name: ")
        resolve_to_concept_id(local_mapping_name, concept_name)
