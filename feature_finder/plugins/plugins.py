class PluginException(Exception):
    pass


def string_length(data):
    """Sample plugin that generates a selector from non-quantitave data."""
    from pandas.api.types import is_string_dtype
    for column in data.columns:
        if is_string_dtype(data[column]):
            data[column] = data[column].str.len()

    return data
