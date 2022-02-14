from algorithm import ProbsAlgo
import argparse


def args_parsing():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_data_path', required=True, help='input data path')

    parser.add_argument('-n', '--n_iterations', type=int,
                        help='number of Monte Karlo iterations',
                        default=100)

    parser.add_argument('-o', '--output_pic_path', type=str, required=True, help='output path')

    parser.add_argument('-p', '--probs', type=float, required=True,
                        nargs=3, help='probabilities of classes')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = args_parsing()
    test = ProbsAlgo(args.input_data_path, args.probs, args.n_iterations)
    test.plot_and_save_result(args.output_pic_path)