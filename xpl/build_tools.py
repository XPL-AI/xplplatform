import os
import shutil


def move_distributions(from_directory: str, library_name: str):
    if not os.path.exists(os.path.join(os.environ['XPL_CODE_DIR'], 'libs')):
        os.mkdir(os.path.join(os.environ['XPL_CODE_DIR'], 'libs'))

    library_directory = os.path.join(os.environ['XPL_CODE_DIR'], 'libs', library_name)
    if not os.path.exists(library_directory):
        os.mkdir(library_directory)

    if os.path.exists(os.path.join(from_directory, 'dist')):
        for file_name in os.listdir(os.path.join(from_directory, 'dist')):
            shutil.move(
                os.path.join(from_directory, 'dist', file_name),
                os.path.join(library_directory, file_name))

    for file_name in os.listdir(from_directory):
        if file_name.endswith(".yaml"):
            shutil.copy(
                os.path.join(from_directory, file_name),
                os.path.join(library_directory, file_name))

    cleanup(from_directory)


def cleanup(directory: str, include_egg_info=False):
    shutil.rmtree(os.path.join(directory, 'build'), ignore_errors=True)
    shutil.rmtree(os.path.join(directory, 'dist'), ignore_errors=True)

    if include_egg_info:
        for file_name in os.listdir(directory):
            if file_name.endswith(".egg-info"):
                shutil.rmtree(os.path.join(directory, file_name), ignore_errors=True)
