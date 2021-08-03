from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--onlyeval', action='store_true')
args = parser.parse_args()
print(args.onlyeval)
