import csv
import math
import os
import sys

LOVED_SONGS = [124,210,0.78,0.82]

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(THIS_FOLDER, 'songs.csv')

def dot(a,b):
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    return sum( x * y for x, y in zip(a,b) )

def magnitude(a):
    return math.sqrt(dot(a, a))

def cosine_similarity(a,b):
    Length_a = magnitude(a)
    Length_b = magnitude(b)
    if Length_a == 0 or Length_b ==0:
        return 0.0
    return dot(a,b)/ (Length_a * Length_b)

def min_max_fit(rows):
    columns = list(zip(*rows))
    low = [min(column) for column in columns]
    high = [max(column) for column in columns]
    return low, high

def min_max_Scale(vector, min, max):
    scaled = []
    for value, low, high in zip(vector, min, max):
        if high == low:
            scaled.append(0.0)
        else:
            scaled.append((value - low) / (high-low))
    return scaled

def load_table(path):
    labels = []
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
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
        marker = " <--- Top 3" if position <= 3 else ""
        print (f" {position:>2}, {label:<22} {score:.6f}{marker}")
    print()

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    labels, rows = load_table(path)

    print_ranking("Without normalization (raw numbers)", rank(LOVED_SONGS, labels, rows))
    low, high = min_max_fit(rows) 
    scaled_rows = [min_max_Scale(row, low, high) for row in rows]
    scaled_song = min_max_Scale(LOVED_SONGS, low, high)

    print_ranking("Using min-max normilation", rank(scaled_song, labels, scaled_rows))  

if __name__ == "__main__":
    main() 
