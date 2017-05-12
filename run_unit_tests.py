from argparse import ArgumentParser
from pprint import pprint
from dev.tests import UnitTest
from dev.ESOINN import EnhancedSelfOrganizingIncrementalNN


def parse_argv():
    parser = ArgumentParser(description="Run unit tests on ESOINN.")
    parser.add_argument('--n_times', '-n', type=int, default=0,
                        help="Number of times to test methods.")
    parser.add_argument('--plot', '-p', type=int, default=0,
                        help="Plot graph [1 - plot before tests, 2 - after,"
                             "3 - both].")
    parser.add_argument('--log', '-l', type=bool, default=False,
                        help="Show logs.")
    parser.add_argument('--state', '-s', type=bool, default=False,
                        help="Show nn state after tests.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argv()
    test_nn = EnhancedSelfOrganizingIncrementalNN()
    unit_test = UnitTest(test_nn)
    if args.plot == 1 or args.plot == 3:
        unit_test.display_info(plot=True)
    success = unit_test.run_tests(args.n_times)
    if args.plot == 2 or args.plot == 3:
        unit_test.display_info(plot=True)
    unit_test.display_info(log=args.log, show=False)
    if args.state:
        pprint(test_nn.current_state(deep=False))
    if not success:
        exit(1)
