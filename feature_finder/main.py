import argparse, os

from feature_finder.find_features import get_model, setup_data, print_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help=('Model type: linear or logistic.'))
    parser.add_argument('data', help=('Data in CSV format.'))
    parser.add_argument('y', help=('y-value of column (index position or header name).'))
    parser.add_argument(
        '--header',
        help='Data CSV contains header row.',
        action='store_true'
    )
    parser.add_argument(
        '--plugins',
        nargs='+',
        help='Use plugin(s) defined in plugins directory to add or clean up feature columns.',
    )
    args = parser.parse_args()

    if not args.data:
        print('Please provide the path to your data (CSV).')
        exit(1)

    data, y = setup_data(args.data, args.header, args.y)

    model = get_model(args.model, args.plugins)
    print_stats(model.select(data, y), args.model)


if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    main()
