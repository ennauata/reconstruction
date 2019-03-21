import numpy as np

# string = """
# a & 1.1 & 2 \\
# b & 2.2 & 3 \\
# """
string = """
Hamaguchi 26.4 26.2 7.3 5.7 46.4 22.9
independent 0 0 0 0 0 0 
maxpool 0 0 0 0 0 0 
fully 0 0 0 0 0 0 
noimage 0 0 0 0 0 0
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
