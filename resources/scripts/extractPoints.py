#!/usr/bin/env python3
import sys
import Data.IO as dio

def main():
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} path/to/Data_dir")
    dio.extractPoints(sys.argv[1])

if __name__ == '__main__':
    main()
