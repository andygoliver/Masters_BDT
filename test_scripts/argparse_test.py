import argparse 

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "a",
    type=str,
    help="First argument: any string.",
)
parser.add_argument(
    "--b",
    type=str,
    default=None,
    nargs = '+',
    choices = ['mean', 'median', 'max', 'min', 'sum'],
    help="List of strings.",
)
parser.add_argument(
    "--c",
    action = "store_true",
    help="Boolean test value."
)
parser.add_argument(
    "--d",
    type=int,
    default=None,
    nargs = '+',
    choices = range(151),
    help="List of ints.",
)
args = parser.parse_args()

print(f'a = {args.a}')
print(f'b = {args.b}')
print(f'c = {args.c}')
print(f'd = {args.d}')

print(type(args.b))

if args.b is not None:
    print(args.b[0])

    for i in range(1, len(args.b)):
        print(args.b[i])

print(type(args.d))

if args.d is not None:
    print(args.d[0])

    for i in range(1, len(args.d)):
        print(args.d[i])

if args.c:
    print("\033[92m"+
          "successful :)"+
          "\033[0m")

else:
    print("\033[91m"+"failed :("+"\033[0m")