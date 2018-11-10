
def reduce_mean(input):
    for dim in range(len(input.shape)-1, 0, -1):
        input = input.mean(dim=dim)
    return input
