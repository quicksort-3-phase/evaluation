#  Quicksort 3-phase

A Python program to visualize the test results from the paper
"Quicksort 3-phase - Parallel sorting on graphics cards programmed in CUDA."

This package consists of the following files:
* visualize.py
* results
    * combination_test_20200126.csv
    * combination_test_20200322.csv
    * combination_test_qs2p_20200118.csv
    * pivot_test_20200404.csv
    * test_20200407.csv
    * test_half_sorted_20200407.csv
    * test_inverse_sorted_20200407.csv
    * test_sorted_20200407.csv
    * test_zero_20200407.csv

### How to run
Usage with the arguments:
```
python3 visualize.py [-h] -i <path to file> [-c <path to file>] [-o <format>]
```
Supported formats:
* eps 
* pdf
* png
* svg

## Authors
* [Joel Bienias](https://github.com/bieniajl) | bieniajl@fius.informatik.uni-stuttgart.de
* [Alexander Fischer](https://github.com/infality/) | st149038@stud.uni-stuttgart.de
* [Rene Richard Tischler](https://github.com/st149535/) | st149535@stud.uni-stuttgart.de
* [Faris Uhlig](https://github.com/farisu) | faris.uhlig@outlook.de
