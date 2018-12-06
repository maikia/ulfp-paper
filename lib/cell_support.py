import helper as hl
import json
import numpy as np
import os
import matplotlib.pylab as plt
import random
import sys
import subprocess
from numpy.lib.recfunctions import append_fields
from eap import field, cell, graph
from neuron import h



def load_swc_segs(dir_morpho, file_morpho, rmv_nodes_with_no_parent=True):
    '''
    Loads the coordinates from the swc file and places them in the recarray as if they were loaded from .hoc (only L
    is not calculated atm)
    :param dir: directory where the file is stored
    :param file: file to be loaded
    :return: recarray with the coordinates
    '''
    # length will not be calculated
    swc_segs = np.loadtxt(os.path.join(dir_morpho, file_morpho))
    names = {0 : 'undefined',
            1 : 'soma',
            2 : 'axon',
            3 : 'basal_dendrite',
            4 : 'apical_dendrite',
            10 : 'structure'}


    swc_recordings = np.zeros(len(swc_segs), dtype=[('idx', 'f4'),
                                                    ('name', 'f4'),
                                                    ('x0', 'f4'),
                                                    ('y0', 'f4'),
                                                    ('z0', 'f4'),
                                                    ('radius', 'f4'),
                                                    ('parent', 'f4')])

    swc_recordings['idx'] = swc_segs[:,0]
    swc_recordings['name'] = swc_segs[:, 1]
    swc_recordings['x0'] = swc_segs[:, 2]
    swc_recordings['y0'] = swc_segs[:, 3]
    swc_recordings['z0'] = swc_segs[:, 4]
    swc_recordings['radius'] = swc_segs[:, 5]
    swc_recordings['parent'] = swc_segs[:, 6]

    no_child_nodes = len(swc_recordings)

    swc_coords = np.zeros(no_child_nodes, dtype=[('x0', 'f4'),
                                                    ('y0', 'f4'),
                                                    ('z0', 'f4'),
                                                    ('x1', 'f4'),
                                                    ('y1', 'f4'),
                                                    ('z1', 'f4'),
                                                    ('L', 'f4'),
                                                    ('radius', 'f4'),
                                                    ('name', 'S40')
                                                    ])

    # shift soma to the (000)
    # make sure there is at least 1 segment soma
    assert len(swc_recordings[swc_recordings['name'] == 1]) > 0, 'there is no soma'
    soma_seg = swc_recordings[swc_recordings['name'] == 1]
    if soma_seg == 1:
        soma_center = [soma_seg['x0'][0],soma_seg['y0'][0],soma_seg['z0'][0]]
    else:
        soma_center = [np.mean(soma_seg['x0']), np.mean(soma_seg['y0']), np.mean(soma_seg['z0'])]
    #shift all the segments by soma center
    swc_recordings['x0'] -= soma_center[0]
    swc_recordings['y0'] -= soma_center[1]
    swc_recordings['z0'] -= soma_center[2]

    # check if soma is the sphere (ie is just one single point)
    idx = 0
    soma_cylindered = False
    if len(swc_recordings[swc_recordings['parent'] <= 0]) > 0:
        # check if any of those segments is soma (if yes, transform it to the cylinder of the same surface)
        if swc_recordings[swc_recordings['parent'] <= 0]['name'] == 1:
            sphere_radius = swc_recordings[swc_recordings['parent'] <= 0]['radius']
            sphere_surface = 4 * np.pi * sphere_radius ** 2
            # cylinder area = 2*pi*r**2 + h(2pi*r) (Neuron uses only cylinder body, no ends?) (assume h = r)
            soma_rh = round(np.sqrt(sphere_surface / (2 * np.pi)),1)

            # now add the segment
            soma_center_x = swc_recordings[swc_recordings['parent'] <= 0]['x0'][0]
            soma_center_y = swc_recordings[swc_recordings['parent'] <= 0]['y0'][0]
            soma_center_z = swc_recordings[swc_recordings['parent'] <= 0]['z0'][0]

            swc_coords[idx]['x1'] = soma_center_x
            swc_coords[idx]['y1'] = soma_center_y-0.5*soma_rh
            swc_coords[idx]['z1'] = soma_center_z
            swc_coords[idx]['L'] = soma_rh
            swc_coords[idx]['radius'] = soma_rh #here it actually takes radius which later is doubled to form the diameter
            swc_coords[idx]['name'] = 'soma'
            swc_coords[idx]['x0'] = soma_center_x
            swc_coords[idx]['y0'] = soma_center_y+0.5*soma_rh
            swc_coords[idx]['z0'] = soma_center_z
            #import pdb; pdb.set_trace()

            idx += 1
            soma_cylindered = True

    for row in swc_recordings[swc_recordings['parent'] > 0]:
        swc_coords[idx]['x1'] = row['x0']
        swc_coords[idx]['y1'] = row['y0']
        swc_coords[idx]['z1'] = row['z0']
        swc_coords[idx]['radius'] = row['radius']
        swc_coords[idx]['name'] = names[row['name']]

        parent = swc_recordings[swc_recordings['idx'] == row['parent']]

        if parent['name'] == 1.0 and soma_cylindered: # check if coords of the parent werent changed
            swc_coords[idx]['x0'] = soma_center_x
            swc_coords[idx]['y0'] = soma_center_y-0.5*soma_rh
            swc_coords[idx]['z0'] = soma_center_z
        else:
            swc_coords[idx]['x0'] = parent['x0']
            swc_coords[idx]['y0'] = parent['y0']
            swc_coords[idx]['z0'] = parent['z0']
        idx += 1

    return swc_coords


def calc_principal_components(data, plotit=False):
    mu = data.mean(axis=0)
    eigenvectors, eigenvalues, V = np.linalg.svd(data.T, full_matrices=False)

    if plotit:
        projected_data = np.dot(data, eigenvectors)
        sigma = projected_data.std(axis=0).mean()
        # draw data points and the principal components
        xData = data[:, 0]
        yData = data[:, 1]
        zData = data[:, 2]

        fig, ax = plt.subplots()
        ax1 = plt.subplot(2,1,1)
        ax1.scatter(xData, yData)
        plt.title('x,y coords')

        ax2 = plt.subplot(2, 1, 2)
        ax2.scatter(yData, zData)
        plt.title('y,z coords')

        for axis in eigenvectors:
            start, end = mu, mu + sigma * axis

            #import pdb; pdb.set_trace()
            ax1.annotate(
                '', xy=end[:2], xycoords='data',
                xytext=start[:2], textcoords='data',
                arrowprops=dict(facecolor='red', width=2.0))
            ax1.set_aspect('equal')

            #import pdb; pdb.set_trace()
            ax2.annotate(
                '', xy=end[1:], xycoords='data',
                xytext=start[1:], textcoords='data',
                arrowprops=dict(facecolor='red', width=2.0))
            ax2.set_aspect('equal')

    return eigenvectors

def rotate_cell(seg_coords,x_angle=0,y_angle=0,z_angle=0):
    # basic rotations (rotation matrix) - rotates around the 0,0,0 coordinate system
    if x_angle == 0 and y_angle == 0 and z_angle == 0:
        return seg_coords

    radians_x = angle_to_radians(x_angle)
    radians_y = angle_to_radians(y_angle)
    radians_z = angle_to_radians(z_angle)

    multiply_x = np.array([[1, 0, 0],[0, np.cos(radians_x), -np.sin(radians_x)],[0, np.sin(radians_x), np.cos(radians_x)]]).round(2)
    multiply_y = np.array([[np.cos(radians_y), 0, np.sin(radians_y)],[0, 1, 0],[-np.sin(radians_y), 0, np.cos(radians_y)]]).round(2)
    multiply_z = np.array([[np.cos(radians_z), -np.sin(radians_z), 0],[np.sin(radians_z), np.cos(radians_z), 0],[0, 0, 1]]).round(2)

    rotated_segs = seg_coords.copy()

    for idx in range(len(rotated_segs)):
        seg_used = rotated_segs[idx]

        seg0 = np.array([[seg_used['x0']], [seg_used['y0']], [seg_used['z0']]])
        seg1 = np.array([[seg_used['x1']], [seg_used['y1']], [seg_used['z1']]])

        # rotate seg0
        rotate0 = np.dot(multiply_x, seg0) # in x axis
        rotate0 = np.matrix(multiply_y)*rotate0 # in y axis
        rotate0 = np.matrix(multiply_z)*rotate0 # in z axis

        # rotate seg1
        rotate1 = np.matrix(multiply_x)*np.matrix(seg1) # in x axis
        rotate1 = np.matrix(multiply_y)*rotate1 # in y axis
        rotate1 = np.matrix(multiply_z)*rotate1 # in z axis

        rotated_segs[idx]['x0'] = rotate0[0,0]
        rotated_segs[idx]['y0'] = rotate0[1, 0]
        rotated_segs[idx]['z0'] = rotate0[2, 0]
        rotated_segs[idx]['x1'] = rotate1[0, 0]
        rotated_segs[idx]['y1'] = rotate1[1, 0]
        rotated_segs[idx]['z1'] = rotate1[2, 0]

    return rotated_segs

def divide_cell_to_layers(seg_coords, params_all):
    layer_ranges = params_all['cell_layer_ranges']
    all_layers = []
    [all_layers.append([]) for i in range(len(layer_ranges))]

    for seg in seg_coords:
        y1, y2 = seg['y0'], seg['y1']
        y = np.mean([y1, y2])

        for idx, next_layer in enumerate(layer_ranges):
            if y < next_layer[1] and y > next_layer[0]:
                all_layers[idx].append(seg)

    return all_layers

def radians_to_angle(radians):
    return radians * 180./np.pi

def angle_to_radians(angle):
    return angle *np.pi/180.

def calc_mean_segs(segs):
    ''' calculates mean coordinate of the given segments'''
    x, y, z = np.mean(segs['x0']), np.mean(segs['y0']), np.mean(segs['z0'])
    return x, y, z

def save_swc_segs(dir_morpho, file_morpho, seg_coords):
    '''
    saves given segs in the rec array format into the .swc file
    :param dir_morpho: directory where the .swc file should be saved
    :param file_morpho: the neme of the new .swc file
    :param seg_coords: seg coords to be saved
    '''
    write_file = os.path.join(dir_morpho, file_morpho)
    swc_file = open(write_file, "w")

    parent_array = np.zeros([len(seg_coords)+1, 4])

    idx = 1
    seg_type = check_seg_type_no('soma')
    radius = seg_coords[0]['radius']
    base = -1

    first_seg = '%d %d %.2f %.2f %.2f %.3f %d \n' % (
        idx, seg_type, seg_coords[0]['x0'], seg_coords[0]['y0'], seg_coords[0]['z0'], radius, base)
    swc_file.write(first_seg)
    parent_array[idx-1] = [idx, seg_coords[0]['x0'], seg_coords[0]['y0'], seg_coords[0]['z0']]

    for seg in seg_coords:
        seg_type = check_seg_type_no(seg_coords[idx-1]['name'])
        radius = seg_coords[idx-1]['radius']
        x, y, z = seg_coords[idx-1]['x1'], seg_coords[idx-1]['y1'], seg_coords[idx-1]['z1']
        x0, y0, z0 = seg_coords[idx-1]['x0'], seg_coords[idx-1]['y0'], seg_coords[idx-1]['z0']
        mask = np.all([(parent_array[:,1]==x0), (parent_array[:,2]==y0), (parent_array[:,3]==z0)], axis=0)
        if sum(mask) > 0:
            base = int((parent_array[mask][0,0]))
        else:
            base = -1
        idx = idx + 1

        next_seg = '%d %d %.2f %.2f %.2f %.3f %d \n' % (idx, seg_type, x, y, z, radius, base)
        swc_file.write(next_seg)

        parent_array[idx - 1] = [idx, x, y, z]
    swc_file.close()

def check_seg_type_no(seg_name):
    if seg_name == 'soma':
        return 1
    elif seg_name == 'axon':
        return 2
    elif seg_name == 'basal_dendrite':
        return 3
    elif seg_name == 'apical_dendrite':
        return 4
    elif seg_name == 'custom':
        return 5
    elif seg_name == 'unspecified_neurites':
        return 6
    elif seg_name == 'glia_processes':
        return 7
    else:
        return 0

def shift_cell(seg_coords, x_shift=0, y_shift=0, z_shift=0):
    # shifts all the coords by the given shifts
    if x_shift == 0 and y_shift == 0 and z_shift == 0:
        return seg_coords
    else:
        seg_coords = seg_coords.copy()

    # shift in x direction
    if x_shift is not 0:
        seg_coords['x0'] = seg_coords['x0'] + x_shift
        seg_coords['x1'] = seg_coords['x1'] + x_shift

    # shift in y direction
    if y_shift is not 0:
        seg_coords['y0'] = seg_coords['y0'] + y_shift
        seg_coords['y1'] = seg_coords['y1'] + y_shift

    # shift in z direction
    if z_shift is not 0:
        seg_coords['z0'] = seg_coords['z0'] + z_shift
        seg_coords['z1'] = seg_coords['z1'] + z_shift
    return seg_coords

def calc_mean_segs(segs):
    ''' calculates mean coordinate of the given segments'''
    x, y, z = np.mean(segs['x0']), np.mean(segs['y0']), np.mean(segs['z0'])
    return x, y, z

def define_coords_for_cells(dir_name, file_name):
    import random as random

    with file(os.path.join(dir_name, file_name), 'r') as fid:
        params_from_file = json.load(fid)

    morpho_dir = str(params_from_file['dir_morpho_vertical'])
    repeat_morpho = params_from_file['repeat_morpho']
    cell_no = params_from_file['cell_no']
    min_cell_distance = params_from_file['min_cell_distance']
    space_x_range  = params_from_file['space_x_range']
    space_y_range = params_from_file['space_soma_range']
    space_z_range  = params_from_file['space_z_range']
    space_x_prob_distribution = params_from_file['space_x_prob_distribution']
    x_hist = params_from_file['x_hist']

    swc_files = hl.find_files(dir_name=morpho_dir, ext='swc')
    if not repeat_morpho:
        assert len(swc_files) >= cell_no and cell_no > 0

    # seclect morphos to work on
    if repeat_morpho:
        # choose randomly morhpologies from the given files (might repeat)
        swc_files = [random.choice(swc_files) for i in range(cell_no)]
    else:
        # choose first cell_no morphologies from the given file list
        swc_files = swc_files[:cell_no]

    coords_array = generate_space_cells(cell_no, min_cell_distance,
                                            space_x_range, space_y_range, space_z_range,
                                        space_x_prob_distribution=space_x_prob_distribution,
                                        x_hist=x_hist)

    try:
        assert cell_no == len(swc_files), 'more cells were created than needed'
    except:
        import pdb;
        pdb.set_trace()
    return swc_files, coords_array

def generate_space_cells(cell_number, min_cell_distance=1.0, 
            x_range=None, y_range=None, z_range=None,
                         space_x_prob_distribution=[1.],
                         x_hist=1000000
                         ):
    """ generates the coordinates for given number of cells,
    making sure that the cells are not closer from each other
    than givem min_cell_distance in microms and that they are covering
    the space range given. If any of the of the dimensions is set to None
    all the coords in this dimension will be set to 0 and the distance will
    not be checked within this dimension """
    space_ranges = [x_range,y_range,z_range]
    
    # check that the given number of cells fits within the span range
    assert check_cells_fit(cell_number, min_cell_distance, space_ranges)
    del space_ranges
    
    # create initial storage arrays           
    coords_array = np.zeros([cell_number, 3])
    
    # works only for x and y axis now; for circular cells
    radius=min_cell_distance*0.5 
    x1_raw = generate_possible_coords(radius,x_range,min_cell_distance)
    x2_raw = generate_possible_coords(min_cell_distance,x_range,min_cell_distance)
    y_space_cell = min_cell_distance/2.*np.sqrt(3.) # from pitagoras
    y_raw=generate_possible_coords(radius,y_range,y_space_cell)
    z_raw = generate_possible_coords(radius,z_range,min_cell_distance)
    
    x1 = True
    all_coords = []
    for next_depth in z_raw:
        if x1 == True:
            x1= False
        else:
            x1 = True
        for next_raw in range(len(y_raw)):
            if x1 == True:
                x1 = False
                for next_coord in range(len(x1_raw)):
                    all_coords.append([x1_raw[next_coord],y_raw[next_raw],next_depth])
            else:
                for next_coord in range(len(x2_raw)):
                    all_coords.append([x2_raw[next_coord],y_raw[next_raw],next_depth])
                x1 = True
            
    # randomly choose the cell coords number which are needed
    from random import choice
    cumsum_layer_syn_prob = np.cumsum(space_x_prob_distribution)
    # normalize
    cumsum_layer_syn_prob = cumsum_layer_syn_prob/np.max(cumsum_layer_syn_prob) # this line was added, might need to be tested for inh neurons

    all_x_layers = np.arange(x_range[0], x_range[1]+x_hist, x_hist)-(0.5*x_hist)
    # first and last 'layer' will have half-width
    all_x_layers[0] = x_range[0]
    all_x_layers[-1] = x_range[1]
    assert len(space_x_prob_distribution) == len(all_x_layers)-1, 'there are '+ str(len(space_x_prob_distribution)) + ' values for probability within x-space, allowed: ' +str(len(all_x_layers)-1)
    for next_cell in range(cell_number):
        all_coords_in_arr = np.array(all_coords)

        # choose how far in x-range
        x = np.random.rand()
        layer_idx = np.searchsorted(cumsum_layer_syn_prob, x)
        layer_idx = np.where(cumsum_layer_syn_prob == cumsum_layer_syn_prob[layer_idx])[0][0]

        '''
        # choose which # here it was always symmetric, let's now change it so the distribution may not be symmetric
        possible = np.where((all_coords_in_arr[:,0] > (x_hist*layer_idx)) & (all_coords_in_arr[:,0] < x_hist*(layer_idx+1)))[0]
        possible_negative = np.where((all_coords_in_arr[:,0] < (-1*x_hist*layer_idx)) & (all_coords_in_arr[:,0] > x_hist*(-1)*(layer_idx+1)))[0]

        possible_all = np.hstack([possible_negative, possible])

        next_choice = choice(possible_all) # possibly there is not enough space for the parameters given to fit all the cells
        '''

        possible = np.where((all_coords_in_arr[:,0] > all_x_layers[layer_idx]) & (all_coords_in_arr[:,0] < all_x_layers[layer_idx+1]))[0]
        next_choice = choice(possible)

        #possible = np.setdiff1d(possible, np.array(next_choice))
        #possible.delete(next_choice)

        coords_array[next_cell] = all_coords[next_choice]
        all_coords.pop(next_choice)

    return coords_array

def check_cells_fit(cell_no, min_cell_distance, space_range=[[0,10],[0,10],None]):
    """ given the number of cells (cell_no), and the minimal distance
    between the cells and the space_ranges (x,y,z) it returns True if the 
    cells can fit within this range and False if not. If any of the 
    dimensions does not exist, type: None"""

    dim1, dim2, dim3 = space_range
    full_dim = 1.
    for dim in [dim1, dim2, dim3]:
        if dim != None:
            dim = dim[1]-dim[0]
            full_dim = full_dim*dim

    return full_dim / min_cell_distance >= cell_no

def generate_possible_coords(starting,a_range,min_cell_distance):
    """ generates possible coords- not working!!!""" 
    a_raw= np.arange(a_range[0]+starting,a_range[1]-starting+1,min_cell_distance)
    
    if len(a_raw) == 0:
        return a_raw
    
    if not check_if_range_filled(a_range,a_raw[-1], min_cell_distance):
        # put one more number on the end if the range is not filled
        a_raw= np.arange(a_range[0]+starting,a_range[1],min_cell_distance) 

    return a_raw

def check_if_range_filled(a_range,last_number, min_cell_distance):
    if a_range == None:
    # check if we are even filling this range
        return True
    elif a_range[1] < last_number+(min_cell_distance*1.5):
    # check if there is no space for more cells
        return True
    else:
        return False

def stim_neuron(general_path, filename_all, filename_cell):
   """ calls the simulating function and passes to it given .json file with the parameters saved """


   current_dir = os.getcwd()
   cmd = [sys.executable, current_dir+'/lib/sim_cell.py',
                     str(general_path),
                     str(filename_all),
                    str(filename_cell)]
   subprocess.call(cmd)


def choose_subset_of_segs(seg_coords, keep_segs = ['apic', 'dend', 'soma']):
    # remove axon and any other unnecessary segs
    idcs = []
    for idx in range(len(seg_coords)):
        last_idx = seg_coords[idx]['name'].find('[')
        if last_idx == -1:
            name_used = seg_coords[idx]['name']
        else:
            name_used = seg_coords[idx]['name'][:last_idx]
        if name_used in keep_segs:
            idcs.append(idx)

    seg_coords_subset = seg_coords[idcs]
    return seg_coords_subset

def add_sections_to_seg_coords(seg_coords):
    ''' adds field 'sections' to the seg_coords rec_array'''
    seg_coords = append_fields(seg_coords, 'section', np.zeros(len(seg_coords)), usemask=False,
                               dtypes=float)  # add new field 'section'

    for seg_name in np.unique(seg_coords['name']):
        all_segs_named = seg_coords[seg_coords['name'] == seg_name]
        nsegs_in_sec = len(all_segs_named)
        sections = [(i * (1. / nsegs_in_sec)) - 0.5 * (1. / nsegs_in_sec) for i in range(1, nsegs_in_sec+1)]
        seg_coords['section'][seg_coords['name'] == seg_name] = sections

    seg_coords = append_fields(seg_coords, 'idx', np.zeros(len(seg_coords)), usemask=False,
                               dtypes=int)  # add new field 'section'
    seg_coords['idx'] = range(len(seg_coords))

    return seg_coords

def place_synapses(layer_syn_prob, next_syn_prob=0.5, range_on_syn_on_one=[1, 2]):
    '''
    :param layer_syn_prob: probability of placing given synapse within the layers
    :param next_syn_prob: probability of placing each next synapse after minimum of synapses were placed
    :param range_on_syn_on_one: minimum and maximum of synapses placed on single postysnaptic neuron
    :return:
    '''
    syn_bylayer = np.zeros(len(layer_syn_prob), dtype='int')
    if sum(layer_syn_prob) != 1:
        layer_syn_prob = (np.array(layer_syn_prob) * (1. / sum(layer_syn_prob))).tolist()

    for syn_no in range(1, range_on_syn_on_one[1]+1):
        # set at least one synapse; use prob for placing or not other synapses
        if range_on_syn_on_one[1] <= 0:
            continue; # do not set any synapse if max synapse is 0
        if (syn_no > range_on_syn_on_one[0]) and (np.random.rand() > next_syn_prob):
            # if the smallest number of synapses is already placed create others with the given probability
            print 'skip this synapse'
            continue; # do not create this synapse (probabilistic)
        else:
            print 'created synapse'
        cumsum_layer_syn_prob = np.cumsum(layer_syn_prob)

        x = np.random.rand() # randomly select the number between 0 and 1.0 to see to which layer this synapse will go
        layer_idx = np.searchsorted(cumsum_layer_syn_prob, x)
        layer_idx = np.where(cumsum_layer_syn_prob == cumsum_layer_syn_prob[layer_idx])[0][0]
        syn_bylayer[layer_idx] += 1
    return syn_bylayer

def add_synapses_bylayer(all_params, cell_params, layers, typ='exc', pt_list=[], moreThanOneSynOnSeg = True):
    """" adds given number of synapses of type. Randomly
    distributing them throughout given layers"""

    if typ == 'exc':
        no_synapses = cell_params['no_exc_syn']
        #syn_e = cell_params['no_exc_syn']
        syn_e = all_params['syn_rev_pot_exc']
        con_weight = all_params['con_weight_exc']
        tau1 = all_params['tau1_exc']
        tau2 = all_params['tau2_exc']
        prop_velocity = all_params['prop_velocity_exc']/(10**(-6)/10**(-3)) # change to um/ms
        tot_no_syn_placed = np.sum(cell_params['no_exc_syn'])
    elif typ == 'inh':
        no_synapses = cell_params['no_inh_syn']
        syn_e = all_params['syn_rev_pot_inh']
        con_weight = all_params['con_weight_inh']
        tau1 = all_params['tau1_inh']
        tau2 = all_params['tau2_inh']
        prop_velocity = all_params['prop_velocity_inh']/(10**(-6)/10**(-3)) # change to um/ms
        tot_no_syn_placed= np.sum(cell_params['no_inh_syn'])
    else:
        assert typ=='exc' or typ == 'inh', 'unknown synapse type'
    stim_delay = all_params['stim_delay']
    cell_layer_ranges = all_params['cell_layer_ranges']
    shift_coords = cell_params['cell_coords']  # shift synapses by the shift of the cell


    for idx_layer in range(len(layers)):
        # check if all of the cell segments are within layers
        try:
            len(layers[idx_layer])
            no_synapses[idx_layer] > 0
        except:
            import pdb; pdb.set_trace()

        if len(layers[idx_layer]) == 0 and no_synapses[idx_layer] > 0:
            import pdb; pdb.set_trace()
            continue;
            #tot_no_syn_placed-=no_synapses[idx_layer]
        try:
            range(no_synapses[idx_layer])
        except:
            import pdb;
            pdb.set_trace()

        for inx_syn in range(no_synapses[idx_layer]):
            sec_place = 0.5

            assert averages_are_in_ranges(np.array(layers[idx_layer]), 'y0', 'y1', cell_layer_ranges[idx_layer])

            while np.round(sec_place,1) == 0.5:
                # select randomly section, but make sure that it is not 0.5 because some dendrites are connected
                # at this location and it creates multiple synapses
                # the following function would be much more efficient if the number of synapses would be selected at once

                #dend_no = generate_rand_from_neuron(rand_range=[0, len(layers[idx_layer]) - 1], distribution="int_uniform")

                assert len(layers[idx_layer]) > 0
                dend_no = draw_dendrite_from_pull(np.array(layers[idx_layer]))

                dend_no = int(dend_no)
                sec_place = layers[idx_layer][dend_no]['section']

                # save it and then remove this dendrite from the recarray.
                sec_selected = layers[idx_layer][dend_no]

                if not moreThanOneSynOnSeg:
                    # if not more than 1 synapse is allowed on a single segment, than remove this segment from possibilities
                    layers[idx_layer] = np.delete(layers[idx_layer], dend_no)

            # find which of the sections is the selected section
            sec = 'h.' + sec_selected['name']

            (x, y, z) = cell.get_locs_coord(eval(sec),sec_place)

            syn_coords = [x + shift_coords[0], y + shift_coords[1],
                          z + shift_coords[2]]  # NEURON does not know about the shift therefore it has to be added

            delay = calculate_delay(pt1=[0, 0, 0], pt2=syn_coords, propagation_velocity=prop_velocity)

            try:
                pt_list = add_exp2syn(section=eval('cell.h.' + sec_selected['name']),
                            sec_place=sec_place, con_weight=con_weight, pt_processes=pt_list,
                            syn_e=syn_e, tau1=tau1, tau2=tau2, con_delay = stim_delay+delay)
            except:
                import pdb;
                pdb.set_trace()
                print 'except in adding synapses'

    return pt_list, tot_no_syn_placed

def averages_are_in_ranges(st_array, key0, key1, ranges):
    if np.any((st_array[key0] + st_array[key1]) / 2. < ranges[0]) or np.any((st_array[key0] + st_array[key1]) / 2. > ranges[1]):
        import pdb; pdb.set_trace()
        return False
    else:
        return True

def draw_dendrite_from_pull(layer):
    # add new field to layer called avg_y - based on this field you will later look at the synapse distribution
    # it divides all the dendrites in this layer to number of bins and then draws from randomly selected bin
    # this makes the distribution of the synapses in the layer more uniform

    avg_y = (layer['y0'] + layer['y1']) / 2.

    layer = append_fields(layer, 'avg_y', np.zeros(len(layer)), usemask=False,
                               dtypes=float)  # add new field 'section' to recarray
    layer['avg_y'] = avg_y

    idcs = np.argsort(avg_y)
    gauss_distr = gaussian(np.linspace(-3., 3., len(avg_y)), 0, 2)

    # add gaussian weight to each of the dendrites (based on it's placement in the y direction)
    layer = append_fields(layer, 'weight', np.zeros(len(layer)), usemask=False,
                          dtypes=float)  # add new field 'section'
    layer['weight'] = gauss_distr[idcs[::-1]]

    no_bins = 15
    counts, binEdges = np.histogram(layer['avg_y'], bins = no_bins)
    bin_placement = np.digitize(layer['avg_y'], binEdges)
    bin_placement[layer['avg_y'] == binEdges[-1]] = no_bins # bins[i-1] <= x < bins[i] so we need to put last element to the last bin, not outside of it
    # randomly select the bin
    dend_bin = random.randint(1, no_bins)
    while counts[dend_bin-1] == 0:
        # make sure that there are some segments in this bin
        print 'selecting new dendritic bin'
        dend_bin = random.randint(1, no_bins)
        if np.sum(counts) == 0:
            import pdb; pdb.set_trace()
            print 'there are not enough segments to place all of the synapses'

    idx_allowed_dendrites, = np.where(bin_placement == dend_bin)
    dend_no = random.choice(idx_allowed_dendrites)

    return dend_no

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def calculate_delay(pt1, pt2, propagation_velocity):
    ''' calculates time delay for propagation of the signal between pt1 and pt2'''
    # calculate distance from coords to the (000)
    dist = dist_between_pts(pt1, pt2)

    # calculate the time delay to reach this point
    delay = dist/propagation_velocity

    return delay

def dist_between_pts(pt1, pt2):
    x1, y1, z1 = pt1
    x2, y2, z2 = pt2
    dist = np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
    return dist

def add_exp2syn(section, sec_place=0.5, stim_no=1, syn_e=0.0,
                stim_interval=1, stim_noise=0,
                con_thresh=0, con_delay=1, con_weight=.25e-3,
                pt_processes=[], tau1 = .1, tau2 = 5.):
    """ add EXP2Syn synapse to the cell and it's stimulation
    and connects the two (stim_: stimulation parameters,
    con_: connection parameters"""
    stimlist = []
    nclist = []

    syn = cell.h.Exp2Syn(sec_place, sec=section)


    syn.e = syn_e
    syn.tau1 = tau1 #.1
    syn.tau2 = tau2 #5

    stimlist.append(cell.h.NetStim())
    stimlist[-1].start=0.1
    stimlist[-1].number=stim_no
    stimlist[-1].interval=stim_interval
    stimlist[-1].noise=stim_noise

    nclist.append(cell.h.NetCon(stimlist[-1],syn,con_thresh,
                            con_delay,con_weight))
    pt_processes.append((syn, stimlist[-1], nclist[-1]))


    return pt_processes


def pts2ms(pts, dt):
    """ converts number of pts to ms given timestep (dt) """
    return pts*dt

def ms2pts(ms, dt):
    """ converts ms to pts given timestep (dt) """
    return int(ms/dt)
