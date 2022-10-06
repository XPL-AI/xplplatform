import prodigy
from prodigy.components.loaders import Audio

# @prodigy.recipe(
#     "my-custom-recipe",
#     dataset=("Dataset to save answers to", "positional", None, str),
#     view_id=("Annotation interface", "option", "v", str)
# )
# def recepie_function():
#
#
#     def update(examples):
#         # This function is triggered when Prodigy receives annotations
#         print(f"Received {len(examples)} annotations!")
#
#     pass

prodigy.serve("audio.manual "
              "my_first_sound "
              "/ "
              "--label male_voice,female_voice",
              port=9000)

# prodigy.get_stream