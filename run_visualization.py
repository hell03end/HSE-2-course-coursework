from argparse import ArgumentParser
from dev.ESOINN import EnhancedSelfOrganizingIncrementalNN
from dev.commons import Plotter
from dev.samples import TrainingSamples


def parse_argv():
    parser = ArgumentParser(description="Run ESOINN training visualization.")
    parser.add_argument('--path', '-p', type=str, default=r"visualization.png",
                        help="Path to buffer image.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argv()
    tsmpl = TrainingSamples(123)
    samples = tsmpl.get_gauss_sample(count=4, bias=1,
                                     size=[5000, 6000, 7000, 3000], noise=0,
                                     sigma=0.4, shuffle=False,
                                     classified=False)
    params = {
        'c1': 0.001,
        'c2': 1,
        'learning_step': 100,
        'max_age': 20,
        'forget': False,
        'strong_period_condition': False,
        'strong_merge_condition': True,  # remove later
        'adaptive_noise_removal': False,
        'logging_level': "info",
        'full_logging_info': False
    }
    nn = EnhancedSelfOrganizingIncrementalNN(**params)
    plotter = Plotter(nn)

    for i, sample in enumerate(samples['samples']):
        nn.partial_fit(sample)
        plotter.save_info(args.path, log=False, equal=True, annotate=False)
    nn.update(remove_noise=True)
    plotter.display_info(plot=True, separate_show=False, log=False,
                         annotate=False)
