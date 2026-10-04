import sys
from deepanepose.data import io as dio

def main():
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} path/to/Data_dir")
    dio.extractPoints(sys.argv[1])

if __name__ == '__main__':
    main()
