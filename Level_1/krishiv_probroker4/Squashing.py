import math

def sigmoid_naive(x):
    return 1 / (1 + math.exp(-x))

def sigmoid(x):
    if x>=0:
        return 1 / (1+math.exp(-x))
    e = math.exp(x)
    return e / (1+e)

def relu(x):
    return max(0, x)

def tanh(x):
    e = math.exp(-2 * abs(x))
    value = (1 - e) / (1 + e)
    return value if x >0 else - value

def main():
    print ("  x sigmoid relu tanh ")
    for x in [-5, -1, 0, 1, 5]:
        print (f"{x:>3} {sigmoid(x):>7.4f} {relu(x):>5.1f} {tanh(x):>6.4f}")
    print()

    breaking_input = -1000
    try:
        sigmoid_naive(breaking_input)
    except OverflowError as error:
        print(f"sigmoid_naive({breaking_input}) broke: OverflowError: {error}")
    print(f"sigmoid({breaking_input}) works: {sigmoid(breaking_input)}")

if __name__ == "__main__":
    main()