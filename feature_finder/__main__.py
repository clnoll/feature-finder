import argparse

from feature_finder.find_features import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help=('Model type: linear or logistic.'))
    parser.add_argument('data', help=('Data in CSV format.'))
    parser.add_argument(
        '--plugins',
        help='Use plugin(s) defined in plugins directory to add or clean up feature columns.',
        action='store_true'
    )
    args = parser.parse_args()
    model_type = args.model
    data_csv = args.data
    plugins = args.plugins

    if not data_csv:
        print("Please provide the path to your data (CSV).")
        exit

    model = Model(model_type, plugins)
    model.select(data_csv)
    model.print_results()


if __name__ == "__main__":
    main()
