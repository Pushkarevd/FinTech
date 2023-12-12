import re

from math_classes import Task, TargetFunc

class Parser:

    def __init__(self, path_file: str):
        self.__path_file = path_file

    def read_file(self):
        with open(self.__path_file, 'r') as file:
            parsed_text = re.split('----.*?----', file.read())[1:]

        target_func = TargetFunc(parsed_text[0])
        matrix = parsed_text[1]
        task = Task(target_func, matrix)
        #print(task.A)

        task.step_algo()


if __name__ == '__main__':
    instance = Parser('./task')
    instance.read_file()