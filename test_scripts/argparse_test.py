import argparse 

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "a",
    type=str,
    help="first argument .",
)
parser.add_argument(
    "--b",
    type=int,
    default=None,
    nargs = '+',
    help="rah",
)
parser.add_argument(
    "--c",
    action = "store_true"
)
args = parser.parse_args()

print(f'a = {args.a}')
print(f'b = {args.b}')
print(f'c = {args.c}')

print(args.b[0])

for i in range(1, len(args.b)):
    print(args.b[i])

if args.c:
    print("\033[92m"+
          "successful :)"+
          "\033[0m")

else:
    print("\033[91m"+"failed :("+"\033[0m")