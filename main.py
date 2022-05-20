import math
from collections.abc import Iterable
from sys import argv


class Rule:
    def __init__(self, probability, from_symbol, to_symbols: list):
        self.to_symbols = to_symbols
        self.from_symbol = from_symbol
        self.probability = probability


class Grammar:
    def __init__(self):
        self.rules = []
        self.unique_states = []

    def addRules(self, rules_file):
        # parse a rule file and add it to the grammar
        with open(rules_file) as rules:
            for rule_txt in rules:
                parts = [rule_part.strip().replace('\n', '') for rule_part in rule_txt.split(' ')]
                prob = float(parts[0])
                from_symbol = parts[1]
                to_symbols = parts[3:]
                rule = Rule(prob, from_symbol, to_symbols)
                self.rules.append(rule)
                symbols = [from_symbol] + to_symbols
                for sym in symbols:
                    if sym not in self.unique_states:
                        self.unique_states.append(sym)

    def get_rules_that_derive_states(self, states: Iterable):
        """
        :param states: list of terminals/non-terminals
        :return: Rules: {r | r = * -> x, x in states}
        """
        states = set(states)
        rules = [rule for rule in self.rules if states.issubset(set(rule.to_symbols))]
        return rules


def main():
    input_grammar = argv[1]  # The name of the file that contains the probabilistic grammar
    input_sentences = argv[2]  # The name of the file that contains the input sentences (tests)
    output_trees = argv[3]  # The name of the output file

    cky_task_str = cky_task(input_grammar, input_sentences)

    print_output(output_trees, cky_task_str)


def cky_task(input_grammar, input_sentences):
    # Create grammar
    grammar = Grammar()
    grammar.addRules(input_grammar)

    # Parse sentences
    sentences_file = open(input_sentences)
    raw_sentences = [sentence for sentence in sentences_file]
    sentences_file.close()
    sentences = [[part.strip().replace('\n', '') for part in sentence.split(' ')] for sentence in raw_sentences]

    output_str = ''
    for sentence in sentences:
        output_str += f'Sentence: {" ".join(sentence)}\n'

        # CKY algo
        n = len(sentence)
        unique_states = grammar.unique_states
        # chart[i][j] is a dict of <state_name> -> probability
        # and also <state_name>_to -> the next step (in order to create the tree)
        chart = [[{} for _ in range(n)] for _ in range(n)]
        for col in range(n):
            # first case -> diagonal
            word = sentence[col]
            rules_that_derive_w = grammar.get_rules_that_derive_states([word])
            for rule in rules_that_derive_w:
                chart[col][col][rule.from_symbol] = rule.probability
                chart[col][col][f'{rule.from_symbol}_to'] = {word: 'terminal'}

            for row in range(col - 1, -1, -1):
                # find the break point that maximise the tree probability
                for break_point in range(row, col):
                    for left_state in [state for state in unique_states if state in chart[row][break_point]]:
                        for right_state in [state for state in unique_states if state in chart[break_point + 1][col]]:
                            join_rules = grammar.get_rules_that_derive_states({left_state, right_state})
                            for join_rule in join_rules:
                                p = join_rule.probability * chart[row][break_point][left_state] * \
                                    chart[break_point + 1][col][right_state]
                                if p > chart[row][col].get(join_rule.from_symbol, -1):
                                    # update the location in chart with the new probability
                                    chart[row][col][join_rule.from_symbol] = p
                                    # Save where we came from
                                    chart[row][col][f'{join_rule.from_symbol}_to'] = \
                                        {left_state: [row, break_point], right_state: [break_point + 1, col]}

        # print output
        output_str += 'Parsing:\n'
        start_pos = chart[0][n - 1]
        if 'S_to' in start_pos:
            output_str += 'S'
            # stack objs are tuples of: (state, location, tabs)
            stack = [(k, v, 1) for k, v in start_pos['S_to'].items()]
            while len(stack) > 0:
                (state, location, tabs) = stack.pop(0)
                if location == 'terminal':
                    output_str += f' > {state}'
                else:
                    output_str += '\n'
                    for _ in range(tabs):
                        output_str += '\t'
                    output_str += state
                    next_states = chart[location[0]][location[1]][f'{state}_to']
                    for k, v in next_states.items():
                        stack.insert(0, (k, v, tabs + 1))

            output_str += f'\nLog probability: {math.log(start_pos["S"])}\n\n'
        else:
            output_str += '*** This sentence is not a member of the language generated by the grammar ***\n\n'

    return output_str


def print_output(output_file, output_str):
    print(f'Writing output to {output_file}')
    with open(output_file, 'w', encoding='utf8') as output_file:
        output_file.write(output_str)


if __name__ == '__main__':
    main()
