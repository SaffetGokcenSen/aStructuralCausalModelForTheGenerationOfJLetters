from generateJ_functions import randomDeviations
from generateJ_functions import drawCap
from generateJ_functions import drawVertical
import numpy as np 
import matplotlib.pyplot as plt
import cv2

# In this part, it is determined which of the samples are to be capped.
# Random number generator defined.
rng1 = np.random.default_rng(seed=1)
# p of J's are to be capped.
p = 0.5
n = 1 
# 3000 J are to be generated.
numOfSamples = 3000
# capped or uncapped 
samples = rng1.binomial(n, p, numOfSamples)
print("histogram of the binomial samples:")
hist, bins = np.histogram(samples)
print(hist)
print(bins)
num_of_capped = hist[-1] 
num_of_uncapped = hist[0]
print("number of capped:")
print(num_of_capped)
print("number of uncapped:")
print(num_of_uncapped)

# The starting points of the approximately horizontal cap 
# m of the starting points 
rng2 = np.random.default_rng(seed=11) 
initial_point_m_lower = 0 
initial_point_m_upper = 3
initial_point_m = rng2.integers(
        initial_point_m_lower, 
        initial_point_m_upper+1, 
        size=numOfSamples)
fig, ax = plt.subplots() 
ax.hist(initial_point_m)
plt.savefig("initial_point_m.png") 

# random n's from the right half of the image are generated
rng3 = np.random.default_rng(seed=21) 
random_right_n = rng3.integers(14, 28, size=numOfSamples) 
# random n's from the left half of the image are generated
rng4 = np.random.default_rng(seed=31) 
random_left_n = rng4.integers(0, 14, size=numOfSamples) 
# cap lengths are generated and their histogram is plotted
cap_lengths = random_right_n -random_left_n + 1
fig, ax = plt.subplots() 
ax.hist(cap_lengths) 
plt.savefig("cap_length2.png") 

variable_E_height = 10 
variable_F_height = 9
variable_F_m_upper_bound = 18
cap_mn_array_list = []
vertical_E_mn_array_list = []
E_mn_array_list = []
F_mn_array_list = []
G_mn_array_list = []

# the ratio of m deviations to be set to zero  
p_zero_dev_m = 0.8
p_zero_dev_n = 0.8 
p_zero_dev_n_F = 0.8
p_zero_dev_n_G = 0.8

# J's are drawn 
for theIndex in range(numOfSamples): 
    sample_initial_point_m = initial_point_m[theIndex] 
    sample_right_n = random_right_n[theIndex] 
    sample_left_n = random_left_n[theIndex] 
    
    # the null image 
    im = np.zeros((28, 28)).astype(int) 

    if (samples[theIndex] == 1):
        sample_cap_length = sample_right_n - sample_left_n + 1 
        seed1 = 41+theIndex 
        size1 = sample_cap_length-1 
        seed2 = 51+theIndex
        random_deviations = randomDeviations(seed1, size1, seed2, p_zero_dev_m) 

        cap_mn_array = np.zeros((sample_cap_length, 2), dtype=int)
        im, cap_mn_array = drawCap(
                im, cap_mn_array, sample_cap_length, sample_initial_point_m, 
                sample_right_n, random_deviations
        ) 
        m_array = cap_mn_array[:, 0]
    else: 
        sample_cap_length = 0 
        cap_mn_array = np.array([0]).astype(int) 

    # the starting point of nearly vertical part of J is determined 
    right_to_left_list = np.arange(sample_left_n, sample_right_n + 1).tolist()  
    vertical_start_lower = np.floor(
            np.quantile(right_to_left_list, 0.25)).astype(int) 
    vertical_start_upper = np.round(
            np.quantile(right_to_left_list, 0.75)).astype(int) 
    rng7 = np.random.default_rng(seed=61+theIndex) 
    vertical_start_n = rng7.integers(vertical_start_lower, vertical_start_upper+1) 
    
    if (samples[theIndex] == 1): 
        vertical_start_m = m_array[sample_right_n-vertical_start_n] + 1 
    else: 
        rng = np.random.default_rng(seed=71+theIndex)
        vertical_start_m = rng.integers(0, 5)
    
    var_E_ver_part_length = variable_E_height-vertical_start_m
    seed1 = 81 + theIndex 
    size1 = var_E_ver_part_length 
    seed2 = 91 + theIndex 
    pZero = p_zero_dev_n
    random_dev = randomDeviations(seed1, size1, seed2, pZero) 
    
    vertical_E_mn_array = np.zeros((var_E_ver_part_length, 2), dtype=int)
    drawnArray = vertical_E_mn_array
    drawActLength = var_E_ver_part_length 
    initial_m = vertical_start_m 
    initial_n = vertical_start_n 
    randomDevs = random_dev
    im, vertical_E_mn_array, current_m, current_n = drawVertical(
            im, drawnArray, drawActLength, initial_m, initial_n, randomDevs)
    
    seed1 = 101 + theIndex 
    ver_start_m_F = variable_E_height 
    var_F_ver_part_length = variable_F_m_upper_bound+1-ver_start_m_F
    size1 = var_F_ver_part_length 
    seed2 = 111 + theIndex 
    pZero = p_zero_dev_n_F
    random_dev = randomDeviations(seed1, size1, seed2, pZero) 

    ver_start_n_F = (np.round(np.mean(vertical_E_mn_array, axis=0)[1])).astype(int) 
    drawnArray = np.zeros((variable_F_height, 2), dtype=int)
    drawActLength = var_F_ver_part_length 
    initial_m = ver_start_m_F 
    initial_n = ver_start_n_F 
    randomDevs = random_dev
    im, vertical_F_mn_array, current_m, current_n = drawVertical(
            im, drawnArray, drawActLength, initial_m, initial_n, randomDevs) 

    # the part in the variable G is to be drawn
    ver_start_n_G = (np.round(np.mean(vertical_F_mn_array, axis=0)[1])).astype(int)
    ver_start_m_G = 19 
    if (samples[theIndex] == 1):
        ver_stop_m_G = (np.round(
            27-0.35*sample_cap_length-0.20*var_E_ver_part_length
        )).astype(int) 
    else: 
        ver_stop_m_G = (np.round(27-0.40*var_E_ver_part_length)).astype(int) 
    
    if (ver_stop_m_G > 27): 
        ver_stop_m_G = 27

    if (ver_stop_m_G > 19):
        var_G_ver_part_length = ver_stop_m_G+1-19 
        seed1 = 121 + theIndex 
        size1 = var_G_ver_part_length
        seed2 = 131 + theIndex
        pZero = p_zero_dev_n_G
        random_dev = randomDeviations(seed1, size1, seed2, pZero)

        vertical_G_mn_array = np.zeros((var_G_ver_part_length, 2), dtype=int)
        drawnArray = vertical_G_mn_array
        drawActLength = var_G_ver_part_length 
        initial_m = ver_start_m_G 
        initial_n = ver_start_n_G 
        randomDevs = random_dev
        im, vertical_G_mn_array, current_m, current_n = drawVertical(
                im, drawnArray, drawActLength, initial_m, initial_n, randomDevs
        )
    else: 
        vertical_G_mn_array = np.array([[19, ver_start_n_G]]).astype(int) 
        current_m = 19 
        current_n = ver_start_n_G

    hook_start_m = current_m + 1
    hook_start_n = current_n + 1 

    # the coordinates of the vertex are determined assuming that the origin is 
    # at the lower left of the image and the image is in the first quadrant of 
    # the x-y plane.
    if (samples[theIndex] == 1):
        v_x = (np.round(
            hook_start_n-(0.25*sample_cap_length+0.1*var_E_ver_part_length)
        )).astype(int) 
    else: 
        v_x = (np.round(
            hook_start_n-(0.3*var_E_ver_part_length)
        )).astype(int) 

    if (v_x < 0): 
        v_x = 0 

    if (samples[theIndex] == 1):
        v_y = (np.round(27-
            (1.0*sample_cap_length+0.5*var_E_ver_part_length))).astype(int)
    else: 
        v_y = (np.round(27-
            (1.0*var_E_ver_part_length))).astype(int)
    
    if (v_y <= hook_start_m): 
        v_y = hook_start_m + 1

    f = (-1)*(hook_start_n-v_x)*(hook_start_n-v_x)/(4*(hook_start_m-v_y)) 
    x_coordinates = np.flip(np.arange(start=v_x, stop=hook_start_n, step=1))
    y_coordinates = (1/(4*f))*(x_coordinates-v_x)*(x_coordinates-v_x)+(27-v_y) 
    y_coordinates = (np.round(y_coordinates)).astype(int)
    hook1_dev_length = hook_start_n - v_x - 1
    hook1_mn_array = np.zeros((x_coordinates.shape[0], 2), dtype=int) 
    for i in range(x_coordinates.shape[0]): 
        current_n = x_coordinates[i] 
        current_m = (27-y_coordinates[i]).astype(int) 
        if (current_m > 27): 
            current_m = 27
        im[current_m, current_n ] = 1.0 
        hook1_mn_array[i, 0] = current_m 
        hook1_mn_array[i, 1] = current_n 

    if (samples[theIndex] == 1):
        hook_stop_n = (np.round(
            v_x-(0.4*sample_cap_length+0.21*var_E_ver_part_length))).astype(int) 
    else: 
        hook_stop_n = (np.round(
            v_x-(0.6*var_E_ver_part_length))).astype(int) 

    if (hook_stop_n <= 0): 
        hook_stop_n = 0 

    x_coordinates = np.flip(np.arange(start=hook_stop_n, stop=v_x, step=1))
    y_coordinates = (1/(4*f))*(x_coordinates-v_x)*(x_coordinates-v_x)+(27-v_y) 
    y_coordinates = (np.round(y_coordinates)).astype(int)
    hook2_mn_array = np.zeros((x_coordinates.shape[0], 2), dtype=int) 
    for i in range(x_coordinates.shape[0]): 
        current_n = x_coordinates[i] 
        current_m = (27-y_coordinates[i]).astype(int) 
        if (current_m > 27): 
            current_m = 27
        im[current_m, current_n ] = 1.0 
        hook2_mn_array[i, 0] = current_m 
        hook2_mn_array[i, 1] = current_n 

    G_mn_array = np.concatenate(
            (vertical_G_mn_array, hook1_mn_array, hook2_mn_array), axis=0) 

    if (samples[theIndex] == 1): 
        E_mn_array = np.concatenate((cap_mn_array, vertical_E_mn_array), axis=0) 
    else: 
        E_mn_array = vertical_E_mn_array

    cap_mn_array_list.append(cap_mn_array) 
    vertical_E_mn_array_list.append(vertical_E_mn_array)
    E_mn_array_list.append(E_mn_array)
    F_mn_array_list.append(vertical_F_mn_array)
    G_mn_array_list.append(G_mn_array) 

    resultsPath = (
    "./generatedJ/"
    ) 
    resultName = 'generatedJ_index_' + str(theIndex) + '.png'
    im_print = im 
    im_print[im_print==1.0] = 255 
    im_print = im_print.astype(np.uint8) 
    # the image data type must be np.uint8
    _ = cv2.imwrite(resultsPath + resultName, im_print) 
