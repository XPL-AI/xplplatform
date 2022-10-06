from google.cloud import firestore

FIRESTORE_CLIENT = firestore.Client()


def check_document_exists(document_id, collection):
    doc_ref = FIRESTORE_CLIENT.collection(collection).document(document_id)
    return doc_ref.get().exists


def create_document(document_id, document, collection):
    doc_ref = FIRESTORE_CLIENT.collection(collection).document(document_id)
    if not doc_ref.get().exists:
        FIRESTORE_CLIENT.collection(collection).document(document_id).set(document)
    else:
        raise FirestoreRepositoryException(f'Conflict. document_id={document_id} collection={collection} already exists.')


def upsert_document(document_id, document, collection):
    FIRESTORE_CLIENT.collection(collection).document(document_id).set(document)


def get_document_by_id(document_id, collection):
    doc_ref = FIRESTORE_CLIENT.collection(collection).document(document_id)
    doc = doc_ref.get()
    if not doc.exists:
        return None

    return doc.to_dict()


def get_documents_by_field(field_name: str, field_value: str, collection, operator: str = u'=='):
    collection_ref = FIRESTORE_CLIENT.collection(collection)

    # if field_value is not None:
    #     field_value = field_value.lower()

    query_ref = collection_ref.where(field_name, operator, field_value)

    documents = query_ref.stream()
    result = []
    for doc in documents:
        result.append(doc.to_dict())

    return result


def get_documents_that_have_field(field_name: str, collection):
    collection_ref = FIRESTORE_CLIENT.collection(collection)

    query_ref = collection_ref.where(field_name, u'>', '!')

    documents = query_ref.stream()
    result = []
    for doc in documents:
        result.append(doc.to_dict())

    return result


class FirestoreRepositoryException(Exception):
    pass
