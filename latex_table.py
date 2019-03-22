import numpy as np

# string = """
# a & 1.1 & 2 \\
# b & 2.2 & 3 \\
# """
string = """
Hamaguchi 26.4 26.2 26.3 7.3 5.7 6.4 46.4 22.9 30.7
independent 78.1 51.6 62.2 45.5 30.8 36.7 31.5 27.1 29.1
maxpool 80.4 50.1 61.8 53.0 32.1 40.0 47.6 32.7 38.8
fully 79.6 52.2 63.1 50.4 33.6 40.3 43.5 32.7 37.3
Ours 82.0 54.3 65.3 51.3 34.9 41.5 49.7 35.5 41.4
"""

def stress(string):
    lines = string.split('\n')
    #all_numbers = [[float(v.strip()) for v in line.strip('\\').split('&')[1:] if v.strip() != ''] for line in lines]
    all_numbers = [[float(v.strip()) for v in line.strip().split(' ')[1:] if v.strip() != ''] for line in lines]

    num = max([len(numbers) for numbers in all_numbers])
    max_numbers = np.array([numbers for numbers in all_numbers if len(numbers) == num]).max(0)
    for numbers, line in zip(all_numbers, lines):
        if len(numbers) != num:
            new_line = line
        else:
            numbers = ['%0.1f'%(number) if number != max_numbers[index] else '\stress{' + '%0.1f'%(number) + '}' for index, number in enumerate(numbers)]
            new_line = "\\hline " + line.split(' ')[0].strip() + ' & ' + ' & '.join(numbers) + ' \\\\'
            pass
        print(new_line)
        continue
    #print(numbers)
    return

stress(string)
