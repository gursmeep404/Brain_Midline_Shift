def calculate_mls(ideal_x, actual_x):
    if actual_x is None:
        return None
    return abs(actual_x - ideal_x)