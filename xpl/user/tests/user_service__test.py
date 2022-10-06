import pytest
import uuid

from xpl.user import user_service
from xpl.user.user_service import UserService, UserAlreadyRegisteredException, UserDoesNotExistException, User, InvalidCredentials

user_service.USER_COLLECTION_NAME = 'autotests_users'
user_service.USER_KEYS_COLLECTION_NAME = 'autotests_user_api_keys'

sut_service = UserService()


def test__create_user__accuracy_scenario():
    email = f'{uuid.uuid4().hex[0:10]}@test.com'
    user = sut_service.create_user(email)

    assert user.email == email
    assert user.username == email
    assert user.user_id is not None
    assert user.api_key is not None
    assert user.datasets_storage_bucket is None

    with pytest.raises(UserAlreadyRegisteredException):
        sut_service.create_user(email)
    datasets_storage_bucket = f'xplai-datasets-{user.user_id}'

    user = sut_service.set_user_datasets_storage_bucket(user_id=user.user_id, datasets_storage_bucket=datasets_storage_bucket)
    user = sut_service.get_user_by_id(user.user_id)

    assert user.datasets_storage_bucket == datasets_storage_bucket

    user_fetched = sut_service.resolve_user(email=email)

    assert user_fetched.dict() == user.dict()


def test__get_user_by_id__temporary_stub_when_user_not_exist():
    non_existing_user_id = str(uuid.uuid4())
    user = sut_service.get_user_by_id(non_existing_user_id)

    assert user is not None
    assert user.user_id == non_existing_user_id
    assert user.api_key is None
    assert user.email == 'stub_user'
    assert user.username == 'stub_user'

    # TODO: use this logic when stub removed
    # with pytest.raises(UserDoesNotExistException):
    #     sut_service.get_user_by_id(str(uuid.uuid4()))


def test__login__accuracy_scenario():
    email = f'{uuid.uuid4().hex[0:10]}@test.com'
    user = sut_service.create_user(email)

    username = user.username
    api_key = user.api_key

    verification_result, _ = sut_service.verify_api_key(username=username, api_key=api_key)
    assert verification_result is True

    with pytest.raises(InvalidCredentials):
        sut_service.verify_api_key(username=username, api_key='fakekey')

    with pytest.raises(InvalidCredentials):
        sut_service.verify_api_key(username='fakeusername', api_key=api_key)

    with pytest.raises(InvalidCredentials):
        sut_service.verify_api_key(username='fakeusername', api_key='fakekey')

    with pytest.raises(InvalidCredentials):
        sut_service.verify_api_key(username=username, api_key=None)

    with pytest.raises(InvalidCredentials):
        sut_service.verify_api_key(username=username, api_key='')

    with pytest.raises(InvalidCredentials):
        sut_service.verify_api_key(username=None, api_key=api_key)

    with pytest.raises(InvalidCredentials):
        sut_service.verify_api_key(username='', api_key=api_key)

    with pytest.raises(InvalidCredentials):
        sut_service.verify_api_key(username=None, api_key=None)

    with pytest.raises(InvalidCredentials):
        sut_service.verify_api_key(username='', api_key='')

    with pytest.raises(InvalidCredentials):
        sut_service.verify_api_key(username='', api_key='')

    with pytest.raises(InvalidCredentials):
        sut_service.verify_api_key(username='same', api_key='same')



