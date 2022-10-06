import time
import pandas
import inquirer


class TextBasedUserInterface:

    def __init__(self):

        self.__last_display_time = time.time()

    def display_results_on_terminal(self, iteration: int, display: pandas.DataFrame):
        
        if time.time() - self.__last_display_time < 1:
            return
        print(''.join(['\n']*100))
        print('=======================')
        print(f'iteration: {int(iteration)}')
        print('-----------------------')
        with pandas.option_context('display.float_format', '{:0.4f}'.format):
            print(display.fillna(''))
        print('-----------------------')
        print(f'Visit \n http://localhost:8097/ for the entire histroy')
        self.__last_display_time = time.time()

    def ask_user_for_execution_id(self, 
                                  all_execution_uuids: list):
        start_fresh_message = 'Start fresh, do not resume from previous runetimes'
        if len(all_execution_uuids) == 0:
            return None
        questions = [
            inquirer.List('execution_uuid',
                          message="Which runtime would you like to resume from?",
                          choices=[start_fresh_message] + all_execution_uuids,
                          default=all_execution_uuids[0]
                          ),
        ]
        answers = inquirer.prompt(questions)
        resume_execution_uuid = answers['execution_uuid']
        if resume_execution_uuid == start_fresh_message:
            return None

        return resume_execution_uuid.split('\t')[0]
