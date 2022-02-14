from algorithm import ProbsAlgo
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_data_path', required=True, help='input data path')

parser.add_argument('-n', '--n_iterations', type=int,
                    help='number of Monte Karlo iterations',
                    default=100)

parser.add_argument('-o', '--output_pic_path', type=str, required=True, help='output path')

parser.add_argument('-p', '--probs', type=float, required=True,
                    nargs=3, help='probabilities of classes')

args = parser.parse_args()

input_data_path = args.input_data_path
n_iterations = args.n_iterations
output_pic_path = args.output_pic_path
probs = args.probs

if sum(args.probs) == 1:
    test = ProbsAlgo(input_data_path, probs, n_iterations)
    test.plot_and_save_result(output_pic_path)

if sum(args.probs) != 1:
    parser.error('Probabilities should sum to 1')




