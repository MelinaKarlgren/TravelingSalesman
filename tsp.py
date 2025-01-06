import numpy as np
import random

def initializeMap(nbOfCities):
    map = np.zeros((nbOfCities, 2))
    # Populate the map with random coordinates
    for i in range(nbOfCities):
        x = np.random.rand()
        y = np.random.rand()
        map[i] = [x, y]
    return map

def swap(map):
    # Select two random indices
    i, j = np.random.choice(len(map), size=2, replace=False)
    # Swap the cities at the chosen indices
    map[i], map[j] = map[j], map[i]
    return map


def two_opt(map):
    i = np.random.randint(0, len(map) - 1)
    j = np.random.randint(i + 1, len(map))  # Ensure j > i

    # Reverse the portion of the route between i and j
    map[i:j + 1] = map[i:j + 1][::-1]  # Simplified reversal (no need for list() and reversed())

    return map


def three_opt(map):
    i, j, k = np.random.choice(len(map), size=3, replace=False)  # Select 3 random indices
    i, j, k = sorted([i, j, k])  # Ensure i < j < k for easier slicing

    # Rearrange sections of the route (reverse between the segments)
    new_map = np.vstack((map[:i], map[i:j + 1][::-1], map[j + 1:k + 1][::-1], map[k + 1:]))

    return new_map

def calc_norm(map):
    tot_norm = 0
    for i in range(len(map) - 1):
        x1 = map[i][0]
        x2 = map[i + 1][0]
        y1 = map[i][1]
        y2 = map[i + 1][1]

        norm = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        tot_norm += norm

    # Calculate the distance between the last and first point to close the loop
    x1 = map[len(map) - 1][0]
    x2 = map[0][0]
    y1 = map[len(map) - 1][1]
    y2 = map[0][1]

    norm = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    tot_norm += norm

    return tot_norm

def simulated_annealing():
    nbOfRuns = 100
    finalLengths = []
    accepted = 0
    for i in range(nbOfRuns):
        nbOfCities=20
        map = initializeMap(nbOfCities)
        random.shuffle(map)
        norm = [calc_norm(map)]

        T_min = 0.0001
        T_initial = 1
        T_list = [T_initial]
        coolRate = 0.98
        it = int(np.log(T_min / T_initial) / np.log(coolRate))
        for i in range(it+1):
            whatmove = np.random.randint(0, 2)
            if whatmove == 0:
                tempMap = swap(map)
            if whatmove == 1:
                tempMap = three_opt(map)
            if whatmove == 2:
                tempMap = two_opt(map)
            tempMapNorm = calc_norm(tempMap)
            error = tempMapNorm - norm[-1]

            if error < 0 or np.random.rand() <= np.exp(-error / T_list[-1]):
                map = tempMap
                norm.append(calc_norm(map))
                accepted += 1
            else:
                norm.append(norm[-1])  # if no change, repeat the last norm value
            T_list.append(T_list[-1]*coolRate)

        finalLengths.append(norm[-1])


def parallel_tempering():
    num_steps = 5000  # How many steps are we simulating for?
    swap_interval = 5

    nbOfCities = 20
    minimum_lengths = []

    nbOfRuns = 100

    for i in range(nbOfRuns):
        finalLengths = []
        maps = []
        temp_order = []
        t_min = 0.1  # Start temperature
        t_max = 1  # End temperature
        n = 50 # Number of temperature points
        temperature_list = [[] for _ in range(n)]

        # Calculate common ratio
        r = (t_max / t_min) ** (1 / (n - 1))

        # Generate temperatures and ensure they do not exceed t_max
        temperatures = np.linspace(t_min, t_max, n).tolist()
        norms = [[] for _ in range(n)]
        for i in range(n):
            maps.append(initializeMap(nbOfCities))
            random.shuffle(maps[i])
            norms[i].append(calc_norm(maps[i]))
            temperature_list[i].append(temperatures[i])
            temp_order.append(i)

        succSwaps = 0
        swap_attempts = 30

        for step in range(num_steps):
            for i in range(n):
                whatmove = np.random.randint(0, 3)
                if whatmove == 0:
                    tempMap = two_opt(maps[i])
                if whatmove == 1:
                    tempMap = three_opt(maps[i])
                if whatmove == 2:
                    tempMap = swap(maps[i])
                tempMapNorm = calc_norm(tempMap)
                error = tempMapNorm - calc_norm(maps[i])

                if error < 0 or np.random.rand() < np.exp(-error / temperatures[i]):
                    maps[i] = tempMap
                    norms[i].append(calc_norm(maps[i]))
                else:
                    norms[i].append(calc_norm(maps[i]))  # if no change, repeat the last norm value


            # Attempt swaps at intervals
            if step % swap_interval == 0:
                succSwaps = 0
                swapped_temps = []  # List of pairs to attempt swaps

                for i in range(swap_attempts):
                    i = random.randint(0, n - 1)
                    index = temp_order.index(i)
                    if index == 0:
                        j = temp_order[1]  # Swap with the next neighbor
                    elif index == len(temp_order)-1:
                        j =  temp_order[-2]  # Swap with the previous neighbor
                    else:
                        # Swap with either the previous or the next neighbor
                        j = temp_order[index+1] if random.random() < 0.5 else temp_order[index-1]

                    index2 = temp_order.index(j)

                    # Compute the swap acceptance probability
                    delta = (norms[temp_order[index2]][-1] - norms[temp_order[index]][-1]) * (1 / temperatures[temp_order[index2]] - 1 / temperatures[temp_order[index]])
                    P_swap = np.exp(delta)

                    # Attempt the swap
                    if np.random.rand() <= P_swap and temp_order[index] not in swapped_temps and temp_order[index2] not in swapped_temps:
                        temperatures[temp_order[index]], temperatures[temp_order[index2]] = temperatures[temp_order[index2]], temperatures[temp_order[index]]  # Swap temperatures

                        temp = temp_order[index]
                        temp_order[index] = temp_order[index2]
                        temp_order[index2] = temp

                        swapped_temps.append(temp_order[index])
                        swapped_temps.append(temp_order[index2])
                        succSwaps += 1

            for i in range(n):
                temperature_list[i].append(temperatures[i])

        for i in range(n):
            finalLengths.append(norms[i][-1])

        minimum_lengths.append(np.min(finalLengths))
