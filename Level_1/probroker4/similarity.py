import csv
import math
import os
import sys

# The song you just loved: tempo (BPM), duration (seconds), energy, danceability.
LOVED_SONG = [124, 210, 0.78, 0.82]

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(THIS_FOLDER, "..", "..", "datasets", "songs.csv")


def dot(a, b):
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    return sum(x * y for x, y in zip(a, b))


def magnitude(a):
    return math.sqrt(dot(a, a))


def cosine_similarity(a, b):
    length_a = magnitude(a)
    length_b = magnitude(b)
    if length_a == 0 or length_b == 0:
        # A zero vector points nowhere, so it is not similar to anything.
        return 0.0
    return dot(a, b) / (length_a * length_b)


def min_max_fit(rows):
    columns = list(zip(*rows))
    mins = [min(column) for column in columns]
    maxs = [max(column) for column in columns]
    return mins, maxs


def min_max_scale(vector, mins, maxs):
    scaled = []
    for value, low, high in zip(vector, mins, maxs):
        if high == low:
            # Every row has the same value here, so there is no range to divide by.
            scaled.append(0.0)
        else:
            scaled.append((value - low) / (high - low))
    return scaled


def load_table(path):
    labels = []
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip the header row
        for record in reader:
            if not record:
                continue
            labels.append(record[0])
            rows.append([float(value) for value in record[1:]])
    return labels, rows


def rank(query, labels, rows):
    scores = [(label, cosine_similarity(query, row)) for label, row in zip(labels, rows)]
    return sorted(scores, key=lambda pair: pair[1], reverse=True)


def print_ranking(heading, ranking):
    print(heading)
    for position, (label, score) in enumerate(ranking, start=1):
        marker = "  <- top 3" if position <= 3 else ""
        print(f"  {position:>2}. {label:<22} {score:.6f}{marker}")
    print()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    labels, rows = load_table(path)

    print_ranking("Without normalization (raw numbers)", rank(LOVED_SONG, labels, rows))

    mins, maxs = min_max_fit(rows)
    scaled_rows = [min_max_scale(row, mins, maxs) for row in rows]
    scaled_song = min_max_scale(LOVED_SONG, mins, maxs)

    print_ranking("With min-max normalization", rank(scaled_song, labels, scaled_rows))


if __name__ == "__main__":
    main()
