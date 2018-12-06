'''
this script is to be called from the server (or from the cmd line). It does all the necessary
calculations for the single cell.
Most importantly it calculates the local field potential but also it might do other actions such as
plot and save figures
'''
import sys
import os
from neuron import h
import numpy as np
import json
import helper as hl

import cell_support as cs

from eap import field, cell, graph

rdm = h.Random()

def set_random_neuron_seed(seed):
    """set seed for random numbers"""
    if seed == 0:
        rdm.MCellRan4()
    else:
        rdm.MCellRan4(seed)

def get_additional_cell_params(cell_params):
    # you can save dendritic length
    # calculate cell surface
    # get resistance of the segments

    def get_seg_resistance(seg_name):
        h("access " + seg_name)
        h("xx = g_pas")
        return 1. / h.xx

    def calculate_surface(diameters, lengths):
        all_segments = []
        # side_surface1 = np.pi * (0.5*diameters[0])**2
        # side_surface2 = np.pi * (0.5*diameters[-1])**2
        for next_seg in range(len(diameters)):
            seg_surface = np.pi * diameters[next_seg] * lengths[next_seg]
            all_segments.append(seg_surface)
        # return side_surface1+side_surface2+np.sum(all_segments)
        return np.sum(all_segments)


    all_sec_l = 0
    for sec in cell.h.allsec():

        # membrane capacitance
        memb_capacitance = sec.cm


        # calculate length of all the segments together
        all_sec_l += sec.L

        cyt_restist = sec.Ra

        # neuron_params = {}

        # neuron_params['mid_dendritic_resistance'] = None # Ohm*cm^2 # same as soma
        # neuron_params['distal_dend_resistance'] = None # Ohm*cm^2 # same as soma
        # neuron_params['soma_input_resistance'] = None # MOhm # calculated elsewhere
        # neuron_params['extracellular_resistivity'] = None #MOhm # set by the user

        cell_params['soma_surface'] = soma_surface  # micrometers^2
        cell_params['soma_memb_resistance'] = soma_resistance  # Ohm*cm^2
        cell_params['dendritic_height'] = dendritic_height  # micrometer
        cell_params['total_dendritic_length'] = all_section_length  # microm
        cell_params['memb_capacitance'] = memb_capacitance  # microF/cm^2 - default 1 Cm
        cell_params['resting_memb_potential'] = memb_pot  # mV
        cell_params['cytoplasmic resistivity'] = cyt_restist  # Ohm*cm - Ra default 35.4
        cell_params['no_segments_one_cell'] = len(first_cell_seg)

def lambda_f(freq, diam, Ra, cm):
    return 1e5 * np.sqrt(diam / (4. * np.pi *freq * Ra * cm))



def init_cell(all_params, cell_params):
    #uses the .hoc template

    # load your model cell
    morpho_dir = all_params['dir_morpho_vertical']
    morpho_file = cell_params['morpho_file']+'.swc' #''.hoc'

    #cell.load_model(str(os.path.join(morpho_dir, morpho_file)))
    cell.load_model_swc(str(os.path.join(morpho_dir, morpho_file)))
    # define number of segments
    for sec in cell.h.allsec():
        if sec.name() == 'soma':
            sec.nseg = 5
        else:
            sec.nseg = max(int((sec.L / (0.1 * lambda_f(100, sec.diam, sec.Ra, sec.cm)) + 0.9) / 2.) * 2 + 1, 5) # set at least 3 segments

    # get coordinates of the segments of your cell(s)
    seg_coords = cell.get_seg_coords()
    return seg_coords


def shift_cell(cell_params, seg_coords):
    # shift cell by the given shift
    x_shift, y_shift, z_shift = cell_params['cell_coords']
    seg_coords = cs.shift_cell(seg_coords, x_shift=x_shift, y_shift=y_shift, z_shift=z_shift)
    # make sure that none of the segments is of 0 length
    return seg_coords


def change_density_to_prob(seg_coords_layers, syn_prob):
    # if given probabilities are given as a density, we multiply it but the total lengths of segments in each layer
    layer_dend_l = []
    for layer_idx in range(len(seg_coords_layers)):
        layer_dend_l.append(np.sum([seg_coords_layers[layer_idx][idx]['L'] for idx in range(len(seg_coords_layers[layer_idx]))]))

    syn_prob = syn_prob * layer_dend_l

    return syn_prob

def choose_synapse_loc(seg_coords_layers, all_params, prob_is_density =False):
    # choose the location of the synapses given the segments in each of the layers
    # if there is no segment in the given layer, do not place the synapse

    # check which layer is empty of any segments
    empty_layer = [idx for idx in range(len(seg_coords_layers)) if len(seg_coords_layers[idx]) == 0]

    inh_syn_prob = np.array(all_params['inh_synapse_prob_layer'])
    exc_syn_prob = np.array(all_params['exc_synapse_prob_layer'])

    if np.sum(inh_syn_prob) > 0:
        inh_syn_prob[empty_layer] = 0.0
        if prob_is_density:
            inh_syn_prob = change_density_to_prob(seg_coords_layers, inh_syn_prob)
        inh_syn_bylayer = cs.place_synapses(inh_syn_prob, all_params['inh_synapse_prob'],
                                            all_params['range_inh_syn_on_one'])  # inhibtiory
    else:
        inh_syn_bylayer = np.zeros(len(inh_syn_prob), dtype='int')

    if np.sum(exc_syn_prob) > 0:
        exc_syn_prob[empty_layer] = 0.0
        if prob_is_density:
            exc_syn_prob = change_density_to_prob(seg_coords_layers, exc_syn_prob)
        exc_syn_bylayer = cs.place_synapses(exc_syn_prob, all_params['exc_synapse_prob'],
                                            all_params['range_exc_syn_on_one'])  # excitatory
    else:
        exc_syn_bylayer = np.zeros(len(exc_syn_prob), dtype='int')

    return inh_syn_bylayer.tolist(), exc_syn_bylayer.tolist()

def init_synapses(all_params, cell_params, seg_coords):
    """ adds the synapses to the cell"""

    keep_segs=['apic', 'dend', 'soma']
    cleaned_seg_coords=cs.choose_subset_of_segs(seg_coords, keep_segs) # segs are already shifted
    cleaned_seg_coords = cs.add_sections_to_seg_coords(cleaned_seg_coords)
    seg_layers = cs.divide_cell_to_layers(cleaned_seg_coords, all_params)

    inh_syn_bylayer, exc_syn_bylayer = choose_synapse_loc(seg_layers, all_params, prob_is_density = False)

    cell_params['no_inh_syn'] = inh_syn_bylayer
    cell_params['no_exc_syn'] = exc_syn_bylayer

    # adds inhibitory and excitatory synapses to the cell
    pt_list, tot_no_syn_placed_exc=cs.add_synapses_bylayer(all_params, cell_params, seg_layers, typ='exc')
    exc_pts = cell.get_point_processes()

    try:
        assert tot_no_syn_placed_exc == len(exc_pts)
    except:
        import pdb; pdb.set_trace()
        print 'more (or less) excitatory synapses were placed than required. This is probably due ' \
              'to the fact that synapse is being placed in the section (probably of the soma) ' \
              'which connects to other segments'
    pt_list, tot_no_syn_placed_inh=cs.add_synapses_bylayer(all_params, cell_params, seg_layers, typ='inh')

    all_pts = cell.get_point_processes()

    inh_pts = [x for x in all_pts if x not in exc_pts]
    try:
        assert tot_no_syn_placed_inh == len(inh_pts)
    except:
        import pdb; pdb.set_trace()
        print 'more inhibitory synapses were placed than required. This is probably due to the fact ' \
        'that synapse is being placed in the section (probably of the soma) ' \
        'which connects to other segments'

    assert len(inh_pts) + len(exc_pts) == len(all_pts) # make sure the inhibitory point process list is calculated correctly
    return cell_params, exc_pts, inh_pts

def get_synapse_coords(cell_params, inh_pts, exc_pts):
    x_shift, y_shift, z_shift = cell_params['cell_coords']
    if len(exc_pts) > 0:
        exc_pts = [cords[1:] for cords in exc_pts]
        exc_pts = np.array(exc_pts)
        exc_pts += [x_shift, y_shift, z_shift]
        # save synapses
        cell_params['exc_syn_coords'] = exc_pts.tolist()
    if len(inh_pts) > 0:
        inh_pts = [cords[1:] for cords in inh_pts]
        inh_pts = np.array(inh_pts)
        inh_pts += [x_shift, y_shift, z_shift]
        # save synapses
        cell_params['inh_syn_coords'] = inh_pts.tolist()
    return cell_params

def get_neuron_synapses():
    synapses = cell.get_point_processes()

    ppt_vecs = []
    for obj, x, y, z in synapses:
        vec = h.Vector()
        vec.record(obj._ref_i)
        ppt_vecs.append(vec)
    return ppt_vecs

def calc_field(all_params, seg_coords,results_sim):
    # run the simulation for t_length
    print 'simulate neuron'
    grid_size = all_params['grid_size']

    external_resistivity = all_params['external_resistivity']

    record_x_range = all_params[
        'record_x_range']  # the grid with defined cells might be of different size that the one where we calculate
    record_y_range = all_params['record_y_range']

    # estimates the field in every defined point of the grid
    print 'estimate the field'
    n_samp = grid_size  # grid size will be n_samp x n_samp
    xx, yy = field.calc_grid(record_x_range, record_y_range, n_samp)  # define grid
    v_ext = field.estimate_on_grid(seg_coords, results_sim['I'], xx, yy, eta=external_resistivity)

    # find dipole moment for this cell
    Q = field.calc_dipole_moment(seg_coords, results_sim['I_axial'])
    Q_len = np.sqrt(np.sum(Q ** 2, 0))

    results_sim = {'xx': xx,  # grid xx
               'yy': yy,  # grid yy
               'Q': Q,  # dipole
               'Q_len': Q_len,  # dipole length
               'v_ext': np.array(v_ext),  # potential
               'seg_coords': seg_coords}

    return results_sim

def get_synapses():
    ppt_vecs = get_neuron_synapses()

    results_syn = {'ppt_vecs': ppt_vecs}
    return results_syn

def simulate_cells(all_params): #synapses, seg_coords_cells, cells, parameters_from_file):
    print 'record what is happening in the synapses'

    timestep = all_params['timestep']
    vec_soma = cell.h.Vector()
    vec_soma.record(eval('h.soma[0]('+str(0.3)+')._ref_v'), sec = eval('h.soma[0]'))  # vector for recording voltage at the soma

    # initialize it with the timestep
    cell.initialize(dt=timestep)  # ms
    cell.h.finitialize(all_params['memb_pot']) # setting initial membrane potential

    # make calculations of I and I axial for each cell separately
    t_length = all_params['sim_len']
    t, I, I_axial = cell.integrate(t_length, i_axial=True)

    results_sim = {'I': I,  # transmembrane current
               'I_axial': I_axial,  # axial current
               'v_soma': np.array(vec_soma)}
    return results_sim

def initiate_all(all_params, cell_params, sim_neuron=True, sim_field=True):

    # initiate cell
    seg_coords = init_cell(all_params, cell_params)

    # shift cell
    seg_coords = shift_cell(cell_params, seg_coords)

    # place the synapses
    # just for testing
    cell_params, exc_pts, inh_pts = init_synapses(all_params, cell_params, seg_coords)

    # get synapse coords
    cell_params = get_synapse_coords(cell_params, inh_pts, exc_pts)

    results = {'seg_coords':seg_coords}
    results.update(get_synapses())
    if sim_neuron:
        results_sim = simulate_cells(all_params)
        results.update(results_sim)

        if sim_field:
            results_sim = calc_field(all_params, seg_coords, results)
            results.update(results_sim)
        else:
            print 'not calculating the field'
    else:
        print 'not simulating the cell'

    # activate the cell and calculate the field
    #results = simulate_cells(all_params, seg_coords)
    # plot various params

    return cell_params, results

if __name__ == '__main__':
    ''' arg1: name of the folder where the general file is store, eg: 'hippocamp'
        arg2: name of the general file (must be of type .json), eg: '2017_4_11_10_32_all.json'
        arg3: name of the cell file to be used, eg: 'c10861.CNG.json'
                (stored as: hippocamp/2017_4_11_10_32_all/cell/params/c10861.CNG.json')
    eg: python sim_cell.py 'hippocamp/' '2017_4_11_16_47_all.json' 'c12866.CNG.json'
    '''
    general_path = sys.argv[1]
    file_name_all = sys.argv[2]
    file_name_cell = sys.argv[3]
    
    sim_neuron =  True
    sim_field = True
    assert (sim_field == False and sim_neuron == False) or sim_neuron == True

    set_seed = False
    if set_seed:
        # set seed for random number generation
        random_seed = 3
        set_random_neuron_seed(random_seed)

    cell_path = os.path.join(general_path, file_name_all[:-5])

    # load parameters
    with file(os.path.join(general_path, file_name_all), 'r') as fid:
        all_params = json.load(fid)

    cell_param_path = os.path.join(cell_path, 'cell/params', file_name_cell)
    with file(cell_param_path, 'r') as fid:
        cell_params = json.load(fid)

    new_cell_params, results = initiate_all(all_params=all_params, cell_params=cell_params,
                                            sim_neuron=sim_neuron, sim_field=sim_field)

    # save results v_ext, dipole, etc
    cell_results_path = os.path.join(cell_path, 'cell/results')
    hl.create_folder(cell_results_path)
    save_file = os.path.join(cell_results_path, file_name_cell[:-5] + '.npz')

    if sim_neuron and sim_field:
        # simulate all
        np.savez(save_file, xx=results['xx'],yy =results['yy'],Q=results['Q'],Q_len=results['Q_len'],
             I=results['I'], I_axial=results['I_axial'],v_ext=results['v_ext'],ppt_vecs=results['ppt_vecs'],
             seg_coords=results['seg_coords'], v_soma=results['v_soma'])
    elif sim_neuron:
        # simulate only neuron but not field
        np.savez(save_file, I=results['I'], I_axial=results['I_axial'],ppt_vecs=results['ppt_vecs'],
             seg_coords=results['seg_coords'], v_soma=results['v_soma'])
    else:
        # only get the structure of the neuron
        np.savez(save_file, ppt_vecs=results['ppt_vecs'], seg_coords=results['seg_coords'])

    # save updated cell_params
    with file(cell_param_path, 'w') as fid:
        json.dump(new_cell_params, fid, indent=True)
    print 'done'

