import uuid

from typing import Optional, List, Dict
from pydantic import BaseModel

from google.cloud import firestore

USER_COLLECTION_NAME = 'users'
USER_KEYS_COLLECTION_NAME = 'user_api_keys'
FIRESTORE_CLIENT = firestore.Client()


class User(BaseModel):
    user_id: str
    email: str
    username: str
    api_key: Optional[str]
    datasets_storage_bucket: Optional[str]


class UserService:
    def __init__(self):
        global FIRESTORE_CLIENT
        if FIRESTORE_CLIENT is None:
            FIRESTORE_CLIENT = firestore.Client()
        self.__firestore_client = FIRESTORE_CLIENT

    def get_user_by_id(self, user_id):
        user_doc = self.__firestore_client.collection(USER_COLLECTION_NAME).document(user_id).get()
        if not user_doc.exists:
            return User(user_id=user_id,
                        email='stub_user',
                        username='stub_user')
            # raise UserDoesNotExistException()

        return User(**user_doc.to_dict())

    def resolve_user(self,
                     username: str = None,
                     email: str = None):
        if not username and not email:
            raise Exception('Either username or email have to be provided')
        if username:
            collection_ref = self.__firestore_client.collection(USER_COLLECTION_NAME)
            query_ref = collection_ref.where('username', '==', username)

            documents = query_ref.stream()
            result = []

            for doc in documents:
                result.append(doc.to_dict())

            if len(result) < 1:
                raise UserDoesNotExistException(f'username={username} was not found')

            if len(result) > 1:
                raise Exception("More that one user found holding same username.")

            return User(**(result[0]))
        if email:
            collection_ref = self.__firestore_client.collection(USER_COLLECTION_NAME)
            query_ref = collection_ref.where('email', '==', email)

            documents = query_ref.stream()
            result = []

            for doc in documents:
                result.append(doc.to_dict())

            if len(result) < 1:
                raise UserDoesNotExistException(f'email={email} was not found')

            if len(result) > 1:
                raise Exception("More that one user found holding same email.")

            return User(**(result[0]))


    def verify_api_key(self, username, api_key):
        if not api_key or not username:
            raise InvalidCredentials
        key_doc = self.__firestore_client.collection(USER_KEYS_COLLECTION_NAME).document(api_key).get()
        if not key_doc.exists:
            raise InvalidCredentials
        elif key_doc.to_dict()['key'] != api_key:
            raise InvalidCredentials
        elif key_doc.to_dict()['username'] != username:
            raise InvalidCredentials

        return True, key_doc.to_dict()

    def create_user(self, email):
        if self.user_already_registered(email):
            raise UserAlreadyRegisteredException()

        user_id = uuid.uuid4().hex
        api_key = uuid.uuid4().hex
        user = {
            'user_id': user_id,
            'email': email,
            'username': email,  # use email for username.
            'api_key': api_key
        }
        user_api_key = {
            'active': True,
            'key': api_key,
            'user_id': user_id,
            'username': email,  # use email for username.
        }

        self.__firestore_client.collection(USER_COLLECTION_NAME).document(user_id).set(user)
        self.__firestore_client.collection(USER_KEYS_COLLECTION_NAME).document(user_api_key['key']).set(user_api_key)

        return self.get_user_by_id(user_id)

    def set_user_datasets_storage_bucket(self, user_id, datasets_storage_bucket):
        user = self.get_user_by_id(user_id)
        user.datasets_storage_bucket = datasets_storage_bucket

        self.__firestore_client.collection(USER_COLLECTION_NAME).document(user_id).set(user.dict())
        return self.get_user_by_id(user_id)

    def user_already_registered(self, email):
        collection_ref = self.__firestore_client.collection(USER_COLLECTION_NAME)
        query_ref = collection_ref.where('email', '==', email)

        documents = query_ref.stream()
        result = []
        for doc in documents:
            result.append(doc.to_dict())

        return len(result) > 0


class UserDoesNotExistException(Exception):
    pass


class UserAlreadyRegisteredException(Exception):
    pass


class InvalidCredentials(Exception):
    pass
