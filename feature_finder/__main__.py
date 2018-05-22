import argparse

from feature_finder.find_features import Model, setup_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help=('Model type: linear or logistic.'))
    parser.add_argument('data', help=('Data in CSV format.'))
    parser.add_argument(
        '--plugins',
        help='Activate plugin(s) defined in plugins directory',
        action='store_true'
    )
    args = parser.parse_args()
    data_csv = args.data
    plugins = args.plugins

    if not data_csv:
        print("Please provide the path to your data (in CSV format).")
        exit

    data = setup_model(data_csv)

    selectors = Model(plugins)
    selectors.select(data)
    selectors.print_results()


if __name__ == "__main__":
    main()
