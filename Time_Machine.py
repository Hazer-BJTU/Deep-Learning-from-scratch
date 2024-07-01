import collections
import re


def read_time_machine(filepath='.\\DataSet\\Time_Machine\\time_machine.txt'):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines_clipped = [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
    return list(filter(lambda item: item != '', lines_clipped))


def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('Error, unknown token: ' + token)


if __name__ == '__main__':
    lines = read_time_machine()
    tokens = tokenize(lines)
    for i in range(50, 60):
        print(tokens[i])
