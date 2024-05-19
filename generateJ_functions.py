import numpy as np 

def randomDeviations(seed1, size1, seed2, pZero): 
    # random deviations are created
    rng = np.random.default_rng(seed=seed1) 
    random_deviations = rng.integers(-1, 2, size=size1)
    # some of the deviations are randomly set to zero
    rng = np.random.default_rng(seed=seed2)
    deviation_indices = np.nonzero(random_deviations)[0] 
    theSize = np.round(deviation_indices.size * pZero).astype(int)  
    no_dev_indices = rng.choice(deviation_indices, replace=False, size=theSize) 
    random_deviations[no_dev_indices] = 0 
    random_deviations = random_deviations.astype(int)
    return random_deviations 

def drawCap(im, drawnArray, capLength, initial_m, initial_n, randomDevs): 
    im[initial_m, initial_n] = 1
    current_m = initial_m 
    current_n = initial_n
    drawnArray[0, 0] = current_m 
    drawnArray[0, 1] = current_n
    for i in range(capLength - 1): 
        next_m = (current_m + randomDevs[i]).astype(int) 
        if (next_m < 0):
            next_m = 0
        next_n = current_n - 1
        drawnArray[i+1, 0] = next_m
        drawnArray[i+1, 1] = next_n
        im[next_m, next_n] = 1
        current_m = next_m
        current_n = next_n 
    return im, drawnArray 

def drawVertical(
        im, drawnArray, drawActLength, initial_m, initial_n, randomDevs): 
    im[initial_m, initial_n] = 1
    current_m = initial_m 
    current_n = initial_n 
    drawnArray[0, 0] = current_m 
    drawnArray[0, 1] = current_n 
    for i in range(drawActLength-1): 
        next_m = current_m + 1 
        next_n = (current_n + randomDevs[i]).astype(int) 
        if (next_n < 0): 
            next_n = 0 
        elif (next_n > 27): 
            next_n = 27 
        im[next_m, next_n] = 1 
        drawnArray[i+1, 0] = next_m 
        drawnArray[i+1, 1] = next_n 
        current_m = next_m 
        current_n = next_n 
    return im, drawnArray, current_m, current_n
