import sys

def main():
    version = sys.argv[1]
    lines = open('MODULE.bazel').read().splitlines()
    for i, l in enumerate(lines):
        if l.startswith('DEFAULT_PYTHON_VERSION = '):
            lines[i] = f'DEFAULT_PYTHON_VERSION = "{version}"'
            break
    with open('MODULE.bazel', 'w') as ouf:
        print(*lines, file=ouf, sep='\n')

if __name__ == '__main__':
    main()
