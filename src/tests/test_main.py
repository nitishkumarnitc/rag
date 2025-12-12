# IMPORTANT! DO NOT MODIFY THIS FILE
import os
import pytest
import pandas as pd
import ast

@pytest.mark.parametrize("csv_file", [os.path.join("src", "FinalResponse.csv")])
def test_validate_csv(csv_file):
    df = pd.read_csv(csv_file)

    # required columns
    assert "top_docs" in df.columns
    assert "response" in df.columns

    # validate content
    for index, row in df.iterrows():
        try:
            top_docs = ast.literal_eval(row["top_docs"])
        except (ValueError, SyntaxError):
            pytest.fail(f"Row {index}: 'top_docs' is not a valid tuple")

        assert isinstance(top_docs, tuple)
        assert isinstance(row["response"], str)
