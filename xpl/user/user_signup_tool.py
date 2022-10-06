import argparse

from xpl.user.user_service import UserService, User, UserAlreadyRegisteredException


OFFSET = '  '


def print_user(usr: User):
    description = f'{OFFSET}user_id: {usr.user_id}\n' \
                  f'{OFFSET}email: {usr.email}\n' \
                  f'{OFFSET}username: {usr.username}\n' \
                  f'{OFFSET}secret: {usr.api_key}\n' \
                  f'{OFFSET}datasets_storage_bucket: {usr.datasets_storage_bucket}\n'
    print(description)


parser = argparse.ArgumentParser(description='XPL New user tool')
parser.add_argument('-e', '--email')
args = parser.parse_args()

email = args.email

user_service = UserService()

try:
    user = user_service.create_user(email=email)

    print(f'\n{OFFSET}Registration complete.\n')

    print_user(user)
except UserAlreadyRegisteredException:
    print(f'\n{OFFSET}Email "{email}" is already registered\n')
    user = user_service.resolve_user(email=email)
    print_user(user)


