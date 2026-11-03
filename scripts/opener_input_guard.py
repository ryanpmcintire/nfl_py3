import numpy as np

CLOSE_INPUT_COLUMNS = frozenset({"spread_line", "spread"})
OPENER_GRADE_COLUMNS = frozenset(
    {"open", "spread_open", "open_home_spread", "tue_open_home_spread", "opener_line"}
)


class CloseInputError(ValueError):
    pass


def assert_opener_input(
    frame, input_col, grade_col, close_columns=CLOSE_INPUT_COLUMNS, allow_close_control=None
):
    if grade_col not in OPENER_GRADE_COLUMNS or input_col == grade_col:
        return
    if input_col not in close_columns or input_col not in frame or grade_col not in frame:
        return
    both = frame[[input_col, grade_col]].dropna().to_numpy(float)
    if both.size == 0 or np.array_equal(both[:, 0], both[:, 1]):
        return
    if allow_close_control:
        print(f"declared close-input control: {allow_close_control}", flush=True)
        return
    n_diff = int((both[:, 0] != both[:, 1]).sum())
    raise CloseInputError(
        f"model input column {input_col!r} is a closing line and differs from the opener grade "
        f"column {grade_col!r} on {n_diff} of {len(both)} rows; build the input from the opener "
        "or pass allow_close_control naming the study as a declared close-input control"
    )
