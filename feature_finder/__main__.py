import argparse, os

from feature_finder.find_features import Model, setup_data


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
        help='Use plugin(s) defined in plugins directory to add or clean up feature columns.',
        action='store_true'
    )
    args = parser.parse_args()
    model_type = args.model
    data_csv = args.data
    header = args.header
    plugins = args.plugins
    y = args.y

    if not data_csv:
        print('Please provide the path to your data (CSV).')
        exit(1)

    data, y = setup_data(data_csv, header, y)

    model = Model(model_type, plugins)
    model.select(data, y)
    model.print_results()


if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    main()
