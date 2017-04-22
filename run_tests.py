from argparse import ArgumentParser
from dev.ESOINN import EnhancedSelfOrganizingIncrementalNN
from dev.tests import CoreTest


def parse_argv():
    parser = ArgumentParser(description="Run unit tests on ESOINN.")
    parser.add_argument('--n_times', '-n', type=int, default=0,
                        help="Number of times to test methods.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argv()
    test_nn = EnhancedSelfOrganizingIncrementalNN([[1, 1], [1, 1]])
    unit_test = CoreTest(test_nn)
    unit_test.initialize_tests()
    unit_test.run_unit_tests(args.n_times)
