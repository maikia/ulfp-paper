import matplotlib.pylab as plt
import numpy as np
from eap import field, cell, graph
import json
import cell_support as cs
import helper as hl
import random as random
import os

def plot_cell_structure(seg_coords, layer_info=None, write_names=True, set_one_color='k', x_range = [-500, 500]):
    '''plot the structure of the given neuron
    :param seg_coords: coords of the given cell
    :param layer_info: layer info should be either none if the neuron is not to be colored or
            it might be colored according to the different layers or segemnts, then this parameter should cosnsist
            of the following params:
            all_layers > can be None, not used by this funct
            colors > color for each seg_coords,
            layer_names > if write_names = True then this will be used for the legend to set the meaning for each color,
                    (as many layer_names as layer_colors)
            layer_ranges > if layers are to be colored it should specify ranges for each of the layers, otherwise it
                    may be set to None
            layer_colors > colors for each layer or for each segment type
    :param write_names: boolean. Either the legend for different colors is set or not
    :param set_one_color: if only one color is to be set; (layer_info must then remain None)
    :param x_range: x_range for the plot
    :return:
    '''
    # extract layer info
    if layer_info is not None:
        all_layers, colors, layer_names, layer_ranges, layer_colors = layer_info

        if all_layers is not None: # coloring by layer
            # adds horizontal lines for the layers
            [plt.hlines(x, x_range[0], x_range[1], 'b', linestyle = '--') for x, y in layer_ranges[1:]]
            if write_names:
                # writes names for each of the layer
                [plt.text(x_range[0]+50, np.mean(layer_ranges[idx]), layer_names[idx],
                  color=layer_colors[idx]) for idx in range(len(layer_ranges))]
        elif write_names: # coloring by segment
            # writes legend for the colors
            1
    else: # one color for all
        colors = np.array([set_one_color for x in range(len(seg_coords))])

    graph.plot_neuron(seg_coords, colors=colors, autolim=True)  # show structure of neuron(s)

def plot_from_segs(seg_coords,color_what=None):
    # make cell and its' segments to be of different colors
    if color_what == 'segs':
        colors, seg_names, colors_used = color_cell(seg_coords, seg_names=[])
        layer_info = None, colors, seg_names, None, colors_used
    elif color_what == 'layers':
        parameters_from_file = {}
        parameters_from_file['layer_ranges'] = [[-500,-20],[-20,30],[30,100],[100,400],[400,1000]]
        parameters_from_file['layer_colors'] = ['r', 'g', 'b', 'm', 'k']
        parameters_from_file['soma_in_layer'] = 's.pyr'
        parameters_from_file['layer_names'] = ['s.o', 's.pyr', 's.l', 's.r.', 's.lm']
        layer_info = cs.get_layer_colors(seg_coords, parameters_from_file)
    else:
        layer_info = None

    plot_cell_structure(seg_coords,layer_info=layer_info, write_names=True)

def color_cell(segments, seg_names = ['axon', 'soma']):
    '''
    define color for different parts of the cell
    :param segments: segments of the cell
    :param seg_names: # e.g. seg_names = ['axon'] - which segments should be colored? If list is empty, then all the
            types of segments will be colored separately. Otherwise listed segments will be colored separately and all
            the others will be colored as 'other' in the same color
    :return: layer_info which is the list of following parameters: colors, seg_names, colors_used
    '''

    # if you want to set specific segment names only, all the others will be plotted in one other color
    seg_others = False
    if not len(seg_names):
        # find all the unique names of the segments(no number of sequence are accounted for)
        temp_list = [name.split('[')[0] for name in
                     np.unique(segments['name'])]  # get only the part of the segment name before '['
        seg_names = (np.unique(temp_list).tolist())

    colors_used = some_colors(number = len(seg_names) + 1) # adds additional color in case some segments are not accounted for (others)
    colors = [] # to store the color for each of the segments

    for seg in segments:
        seg_name = seg['name'].split('[')[0]
        try:
            idx = seg_names.index(seg_name)
        except:
            idx = -1
            seg_others = True

        colors.append(colors_used[idx])
    # if other segments were found than the given ones, they will all be marked as 'other' and given separate color
    if seg_others:
        seg_names.append('other')
    else:
        colors_used = colors_used[:-1]
    return colors, seg_names, colors_used

def some_colors(number = 5):
    """ it returns list of given number of colors"""
    import colorsys
    N = number
    HSV_tuples = [(x*1.0/N, 1.0, 1.0) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

    # if only one color is required don't put in in the list
    if number == 1:
        RGB_tuples = RGB_tuples
    return RGB_tuples

def plot_rotate_morpho(seg_coords,title='',ax=None,color_what=None):
    ''' plots given cell in four different rotations along y axis '''

    # add vertical lines and degrees of the turn
    plt.vlines(0, -500, 1000, linestyles='--', color='r')
    plt.vlines(-600, -500, 1000, linestyles='--', color='r')
    plt.vlines(600, -500, 1000, linestyles='--', color='r')
    plt.vlines(1200, -500, 1000, linestyles='--', color='r')

    plt.text(0, 1020, '90$^\circ$')
    plt.text(-600, 1020, '0$^\circ$')
    plt.text(600, 1020, '180$^\circ$')
    plt.text(1200, 1020, '270$^\circ$')

    # normal translation
    seg_coords0 = cs.shift_cell(seg_coords=seg_coords,x_shift=-600)
    plot_from_segs(seg_coords0, color_what=color_what)

    # rotated by 90 degrees
    seg_coords1 = cs.rotate_cell(seg_coords=seg_coords, x_angle=0, y_angle=90, z_angle=0)
    plot_from_segs(seg_coords1, color_what=color_what)

    # rotated by 180 degrees
    seg_coords2 = cs.rotate_cell(seg_coords, x_angle=0, y_angle=180, z_angle=0)
    seg_coords2 = cs.shift_cell(seg_coords=seg_coords2, x_shift=600)
    plot_from_segs(seg_coords2, color_what=color_what)

    # rotated by 270 degrees
    seg_coords3 = cs.rotate_cell(seg_coords, x_angle=0, y_angle=270, z_angle=0)
    seg_coords3 = cs.shift_cell(seg_coords=seg_coords3,x_shift=1200)
    plot_from_segs(seg_coords3, color_what=color_what)

    plt.title(title)

''' support for plotting functions '''
def clean_plot(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_frame_on(True)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('',fontsize=18)
    plt.ylabel('',fontsize=18)

def plot_memb_current_for_cell(time_pt, params,
                               plot_morpho=False, plot_field = True,
                               plot_synapses=False, plot_current=False, ax=None):


    # this is used for frames of the movie

    v_ext, xx, yy, seg_coords, x_range, y_range, dt = params
    #v_ext, xx, yy, seg_coords, x_range, y_range, inh_syn_coords, exc_syn_coords, dt = params

    max_field = np.max(v_ext)
    if max_field >= 1000:
        # convert to microvolts
        v_ext = v_ext/10e2 # change to microvolts
        scale_type = 'micro'
        max_field/=10e2
    else:
        scale_type = 'nano'

    #v_ext = v_ext / 10e2 # change nano to micro volts
    import matplotlib.colors as colors

    if ax == None:
        ax = plt.subplot(1, 1, 1)

    # draw field
    mycmap, colormap_range_ceil, aspect, one_forth_colormap, colormap_range = helper_provide_imshow_details(v_ext,
                                                                                                            x_range,
                                                                                                         y_range)

    if plot_field:
            pcm = plt.imshow(v_ext[time_pt, :, :], interpolation="nearest",
                         #norm=colors.SymLogNorm(linthresh=0.01 * np.max(v_ext),
                         #                       linscale=1.0,
                         #                       vmin=colormap_range_ceil[0], vmax=colormap_range_ceil[1]),
                         origin='lower',
                         aspect=0.8,

                         extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
                         cmap=mycmap)
            plt.clim(colormap_range[0], colormap_range[1])


    if plot_morpho:
        # draw morpho
        import pdb; pdb.set_trace()
        col = graph.plot_neuron(seg_coords, colors='k', autolim=True)

        soma_idcs, = np.where(seg_coords['name'] == 'soma')

        draw_soma(ax, x0=seg_coords[soma_idcs[0]]['x0'], x1=seg_coords[soma_idcs[-1]]['x1'],
              y0=seg_coords[soma_idcs[0]]['y0'], y1=seg_coords[soma_idcs[-1]]['y1'],
              color='k')
        plt.xlim(x_range)
        plt.ylim(y_range)

    x_tic_label, xtics, y_tic_label, ytics = helper_define_tick_space(x_range, y_range)
    ax.set_yticks(ytics)
    ax.set_yticklabels(y_tic_label)
    ax.set_xticks(xtics)
    ax.set_xticklabels(x_tic_label)

    if plot_field:
        cbar = plt.colorbar(pcm, extend='both', drawedges=False)  # ax=ax[0],
        cbar.set_ticks([colormap_range[0], -one_forth_colormap, 0, one_forth_colormap, colormap_range[1]])
        cbar.set_ticklabels(
            [str(colormap_range[0]), str(-one_forth_colormap), '0', str(one_forth_colormap), colormap_range[1]])

    if scale_type == 'micro':
        cbar.set_label(r"voltage ($\mu$V)", fontsize=18)
    elif scale_type == 'nano':
        cbar.set_label(r"voltage (nV)", fontsize=18)
	#cbar.set_label(r"voltage ($\mu$V)", fontsize=18)
        #cbar.set_label(r"voltage (nV)", fontsize=18)
    cbar.ax.tick_params(labelsize=16)


    if plot_current:
        # draw streamplots
        U = -np.diff(v_ext[time_pt, :, :], axis=0)[:, :-1]
        V = -np.diff(v_ext[time_pt, : ,:], axis=1)[:-1, :]

        plt.streamplot(xx[0, :-1], yy[:-1, 0], V, U, density = 1.0, color = 'g')

    plt.title('time: '+ str(("%0.2f" % (time_pt*dt)))+ 'ms')

    ax.set_xlabel(r"space ($\mu$m)")
    ax.set_ylabel(r"space ($\mu$m)")

    clean_plot(ax)
    plt.tight_layout()
    return mycmap, colormap_range

def helper_provide_imshow_details(v_ext, x_range, y_range):

    # make logarithmic scale from both sides
    mycmap = plt.cm.get_cmap('RdYlGn') #''RdYlGn') #('RdYlGn_r') #('coolwarm') #get colormap
    # 'jet' - use to compare CSD with existing papers,

    colormap_range = np.max([np.max(v_ext),np.abs(np.min(v_ext))])
    colormap_range_ceil = (-colormap_range, colormap_range)

    # aspect of the colormap
    if np.sum(x_range) > 500:
        aspect = 5.9
    elif np.sum(np.abs(y_range)) > 500:
        aspect = 0.2
    else:
        aspect = 0.8

    if colormap_range > 100:
        colormap_range = ([int(-colormap_range/100.0)*100, int(colormap_range/100.0)*100])
    elif colormap_range > 10:
        colormap_range = ([int(-colormap_range/10.0)*10, int(colormap_range/10.0)*10])
    else:
        colormap_range = ([int(-colormap_range), int(colormap_range)])
    #colormap_range = [-200,200]

    one_forth_colormap = int(np.floor(colormap_range[1]/4.))

    return mycmap, colormap_range_ceil, aspect, one_forth_colormap, colormap_range

def helper_define_tick_space(x_range, y_range):
    if sum(x_range) > 100:
        tick_space_x = 100
    else:
        tick_space_x = 20
    if sum(y_range) > 100:
        tick_space_y = 100
    else:
        tick_space_y = 20

    ytics = [-idx for idx in range(0, abs(y_range[0]), tick_space_y)][::-1] + [idx for idx in range(0, abs(y_range[1]), tick_space_y)][1:]
    if int(min(ytics)) == 0:
        y_tic_label = ['0']+ ['']*int(np.abs((ytics[-1]/tick_space_y))-1.) +[str(max(ytics))]
    else:
        y_tic_label = [str(min(ytics))] + ['']*int(np.abs((ytics[0]/tick_space_y))-1.) + ['0']+ ['']*int(np.abs((ytics[-1]/tick_space_y))-1.) +[str(max(ytics))]

    xtics = [-idx for idx in range(0, abs(x_range[0]), tick_space_x)][::-1] + [idx for idx in range(0, int(abs(x_range[1])), tick_space_x)][1:]
    if int(min(xtics)) == 0:
        x_tic_label = ['0']+ ['']*int(np.abs((xtics[-1]/tick_space_x))-1.) +[str(int(max(xtics)))]
    else:
        x_tic_label = [str(int(min(xtics)))] + ['']*int(np.abs((xtics[0]/tick_space_x))-1.) + ['0']+ ['']*int(np.abs((xtics[-1]/tick_space_x))-1.) +[str(int(max(xtics)))]

    return x_tic_label, xtics, y_tic_label, ytics

def plot_electrode_locs(ax, v_ext, y_values, y_range, x_value = 0, no_y_values = 24):

    for idx, y_value in enumerate(y_values):
        y_location = (y_value/v_ext.shape[1])*np.diff(y_range)+y_range[0]

        #ax1.plot(x_value, y_location, '*', color = colors[idx], ms = 10)
        ax.plot(x_value, y_location, '*', color = 'k', ms = 8, mfc="None")

        if idx%5 == 0:
            ax.text(x_value + 10, y_location - 20, str(idx), color='0.2')

def plot_traces(ax, v_ext, dt, mycmap, colormap_range,layers,
                            x_range, y_range, y_values, x_value = -100, get_params = True,
                            final_figs_dir='', ext='.pdf'):
    #max_time = 10
    max_time = cs.pts2ms(v_ext.shape[0], dt)
    timeline = np.linspace(0, max_time, v_ext.shape[0])

    x_value_xaxis = x_value - x_range[0]
    squeeze = (x_range[1]-x_range[0])/v_ext.shape[2]
    x_value_vext = x_value_xaxis/squeeze

    max_field = np.max(np.abs(v_ext))
    if max_field >= 1000:
        # convert to microvolts
        v_ext = v_ext/10e2 # change to microvolts
        scale_type = 'micro'
        max_field/=10e2
    else:
        scale_type = 'nano'
    scale = int(np.ceil(max_field))
    # use smaller scale

    all_loc = np.zeros(len(y_values))

    abs_max_all = np.zeros(len(y_values))
    if get_params:
        time_to_peaks, psp_ampls, idxs = [],[],[]
        time_to_mins, time_to_maxs, start_times = [],[], []


    for idx, y_value in enumerate(y_values):
        #y_location = (y_value/v_ext.shape[1])*np.diff(y_range)+y_range[0]

        try:
            trace = v_ext[:len(timeline),y_value, x_value_vext]
        except:
            import pdb; pdb.set_trace()
        #ax1.plot(x_value, y_location, '*', color = colors[idx], ms = 10)

        loc = (idx+1) * scale # *0.5 exc

        abs_max = max(trace.min(), trace.max(), key=abs)  # get absolute max element
        ax.hlines(y=loc,xmin=0, xmax=timeline[-1] , color = '0.7')

        # normalize abs_max to between 0 and 1 were ranges are colormap_range
        mapped_value = normalize_color_value(abs_max, colormap_range)

        ax.plot(timeline[:], trace + loc, color='k', alpha=0.7, lw=2)
        ax.plot(timeline[:], trace+loc,  color = mycmap(mapped_value), alpha = 0.7,lw=2) #'k')#color = colors[idx])

        if idx%5 == 0:
            ax.text(timeline[-1], loc-0.8, str(idx), color = '0.2')

        abs_max = max(trace.min(), trace.max(), key=abs) # get absolute max element
        abs_max_all[idx] = abs_max
        all_loc[idx] = loc

        if get_params:
            # we are here searching for the local mins and maxs and getting the first one which appears as peak of the trace
            #abs_trace = np.abs(trace)
            #local_minmax = (np.r_[True, abs_trace[1:] > abs_trace[:-1]] & np.r_[abs_trace[:-1] > abs_trace[1:], True])
            #peak_pt = np.where(local_minmax)[0][0]
            peak_pt = np.argmax(np.abs(trace))
            peak_pt_min = np.argmin(trace)
            peak_pt_max = np.argmax(trace)

            psp_start = np.where(trace != 0)[0][0]
            time_to_peak = (peak_pt - psp_start)*dt
            psp_ampl = trace[peak_pt_max] - trace[peak_pt_min]
            ax.fill_between(timeline[psp_start:peak_pt], loc, trace[psp_start:peak_pt]+loc, color='w')
            time_to_peaks.append(time_to_peak)
            psp_ampls.append(psp_ampl)
            idxs.append(idx)
            time_to_mins.append(peak_pt_min*dt)
            time_to_maxs.append(peak_pt_max * dt)
            start_times.append(psp_start*dt)

            #print 'electrode: ', idx, ' t2min:',peak_pt_min*dt,', t2max:', peak_pt_max * dt,', start_t: ',psp_start*dt, ', max ampl: ', trace[peak_pt_max],', min ampl: ', trace[peak_pt_min], 'p2p: ', psp_ampl, 't2argmax:', time_to_peak

    new_cords = [scale, ((len(y_values)) * scale)]

    if layers != None:

        min_electr_possible = translate(0,4,95,0,len(y_values)-1)
        max_electr_possible = translate(100,4,95,0,len(y_values)-1)
        min_electr_in_space = translate(0,min_electr_possible,max_electr_possible,y_range[0],y_range[1])
        max_electr_in_space = translate(len(y_values)-1,min_electr_possible,max_electr_possible,y_range[0],y_range[1])

        y_scale_space = [translate(lay, min_electr_in_space, max_electr_in_space, new_cords[0], new_cords[-1]) for lay
                         in np.array(layers)[:, 0]]
        draw_layers(ax, y_scale_space, timeline)

    #plt.ylim([-5, scale*25])
    plt.ylim(new_cords)

    ylims = ax.get_ylim()

    ticks = np.arange(ylims[0], ylims[1]+scale, scale)
    ticklabels = ['' for idx in ticks]
    ticklabels[2] = 0
    ticklabels[3] = scale
    plt.yticks(ticks, ticklabels)

    props = dict(color='black', linewidth=2, markeredgewidth=0.5)

    ax.axison = False
    make_yaxis(ax, -0.5, offset=0.7, label_txt = ticklabels, ylims = [ticks[2], ticks[3]],**props)

    props = dict(color='black', linewidth=2, markeredgewidth=0.5)
    make_xaxis(ax, yloc=ylims[0], offset=2, **props)


    plt.title('electrode recordings')
    ax.set_xlabel('time (ms)')
    ax.set_xlim([timeline[0], timeline[-1]])
    clean_plot(ax)
    plt.title('x: ' + str(x_value) + 'um, scale: '+scale_type+'V')

    plt.savefig(os.path.join(final_figs_dir, 'trace_' + str(x_value)) + ext)

    if get_params:

        plt.figure()
        ax = plt.subplot(1, 2, 1)

        plt.plot(start_times, idxs, '*',lw=2, label='start',)

        y_axis = v_ext[:, :, x_value_vext]
        min_all = np.argmin(y_axis, 0)
        max_all = np.argmax(y_axis, 0)
        abs_max_all = np.argmax(np.abs(y_axis),0)

        #in_electr = np.linspace(0, len(idxs)-1, len(min_all)) # zle

        ar_space2electr(0, [0,np.size(v_ext,2)], y_values)
        #import pdb; pdb.set_trace()
        #in_electr = np.linspace(ar_space2electr(0,y_values), ar_space2electr(np.size(v_ext,2), y_values), len(min_all))

        space_min_in_electr = translate(0., 4., 95., 0., 19.)
        space_max_in_electr = translate(100., 4., 95., 0., 19.)
        in_electr = np.linspace(space_min_in_electr, space_max_in_electr, len(min_all))

        plt.plot(abs_max_all * dt, in_electr, 'o', lw=3, label='abs max')
        plt.plot(min_all*dt, in_electr, '.', lw=2, label='min')
        plt.plot(max_all*dt, in_electr, '.', lw=2, label = 'max')
        assert len(np.unique(start_times)) == 1 # make sure that all the start time of the psp are the same; if not
        # it's not a problem but different procedure will be needed
        time_to_largest_peak = abs_max_all*dt - start_times[0]

        #print str(x_value), 'um : ', time_to_largest_peak, ', max: ', np.max(time_to_largest_peak), ', min:', np.min(time_to_largest_peak)

        plt.legend()
        plt.title('x: ' + str(x_value) + 'um'+',scale: '+scale_type+'V')
        plt.ylabel('electrode number')
        plt.xlabel('time (ms)')
        plt.ylim([space_min_in_electr, space_max_in_electr])
        if layers != None:

            np.size(v_ext,1)
            y = [translate(lay, -500, 800, min_electr_possible, max_electr_possible) for lay in np.array(layers)[:, 0]]
            y_scale_space
            #y = np.array(layers)[:,0]
            #y = ((y-y[0])*(1.0)/(np.array(layers)[-1,-1]-y[0]))*idxs[-1]
            draw_layers(ax, y, timeline)

        ax2 = plt.subplot(1, 2, 2)
        place = np.linspace(0,np.size(y_axis,1)-1, np.size(y_axis,1))
        place = place.astype(int)
        peak_to_peak = y_axis[max_all,place] + y_axis[min_all,place]

        plt.vlines(0, space_min_in_electr, space_max_in_electr, colors = '0.5', linestyle='-')
        plt.plot(peak_to_peak, in_electr, 'o', lw=2, label='peak_to_peak')
        plt.title('peak to peak deflection')
        plt.xlabel('deflection '+ '('+scale_type+'V)')
        plt.ylim([space_min_in_electr, space_max_in_electr])
        if layers != None:
            deflection_axis = np.linspace(np.min(peak_to_peak), np.max(peak_to_peak), len(y))
            draw_layers(ax2, y, deflection_axis)
            plt.savefig(os.path.join(final_figs_dir, 'trace_params' + str(x_value)) + ext)

        plt.figure()
        ax = plt.subplot(111)
        #import pdb; pdb.set_trace()
        plt.boxplot(np.array(time_to_peaks), patch_artist = True)
        plt.ylabel('time to absoulte peak (ms)')
        plt.ylim([0,4])
        plt.savefig(os.path.join(final_figs_dir, 't2peak_boxplot' + str(x_value)) + ext)

def normalize_color_value(abs_max, colormap_range, use_log=False):
    if abs_max <= colormap_range[0]:
        return 0.
    elif abs_max >= colormap_range[1]:
        return 1.
    else:
        mapped_value = (abs_max - colormap_range[0])/(colormap_range[1] - colormap_range[0])
        return mapped_value


def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def  draw_layers(ax, layers, timeline):
    for layer in layers[1:]:
        ax.hlines(y=layer, xmin=timeline[0], xmax=timeline[-1], linestyles='--')

def make_yaxis(ax, xloc=0, offset=0.05, label_txt = [], ylims = [], **props):
    import matplotlib.lines as lines

    ymin, ymax = ax.get_ylim()
    locs = [loc for loc in ax.yaxis.get_majorticklocs()
            if loc >= ymin and loc <= ymax]
    tickline, = ax.plot([xloc]*len(locs), locs, linestyle='',
                        marker=lines.TICKLEFT, **props)
    axline, = ax.plot([xloc, xloc], [ylims[0], ylims[1]], **props)
    tickline.set_clip_on(False)
    axline.set_clip_on(False)

    for idx, loc in enumerate(locs):
        if len(label_txt) == len(locs):
            ax.text(xloc - offset, loc, label_txt[idx],
                verticalalignment='center',
                horizontalalignment='right')
        else:
            ax.text(xloc - offset, loc, '%1.1f' % loc,
                verticalalignment='center',
                horizontalalignment='right')


def make_xaxis(ax, yloc, offset=0.05, **props):
    import matplotlib.lines as lines
    xmin, xmax = ax.get_xlim()
    xmin = 0
    locs = [loc for loc in ax.xaxis.get_majorticklocs()
            if loc >= xmin and loc < xmax]
    tickline, = ax.plot(locs, [yloc]*len(locs), linestyle='',
                        marker=lines.TICKDOWN, **props)
    axline, = ax.plot([xmin, xmax-1], [yloc, yloc], **props)
    tickline.set_clip_on(False)
    axline.set_clip_on(False)
    for loc in locs:
        ax.text(loc, yloc - offset, str(int(loc)), #''%1.1f' % loc,
                horizontalalignment='center',
                verticalalignment='top')

def ar_space2electr(x_inSpace, space_range, el_loc):
    '''
    Convert unknown x in space into the
    :param x_inSpace:
    :param el_loc:
    :return:
    '''
    x_electr = (x_inSpace - el_loc[0]) / (el_loc[-1] - el_loc[0]) * len(el_loc)
    return x_electr

def plot_synapse_dist(inh_synapses, exc_synapses,x_range='auto',y_range='auto',layers=None):
    axScatter, axHistx, axHisty = None, None, None
    if len(inh_synapses)> 0:
        inh_synapses = np.array(inh_synapses)
        axScatter, axHistx, axHisty = plot_synapse_distribution(inh_synapses[:,0], inh_synapses[:,1], color='r',x_range=x_range,y_range=y_range, layers=layers)
        plt.title('inh syn ='+str(len(inh_synapses)))
    if len(exc_synapses)>0:
        exc_synapses = np.array(exc_synapses)
        axScatter, axHistx, axHisty = plot_synapse_distribution(exc_synapses[:,0], exc_synapses[:,1], color='g',x_range=x_range,y_range=y_range, layers=layers)
        plt.title('exc syn ='+str(len(exc_synapses)))
    return axScatter, axHistx, axHisty

def plot_synapse_distribution(x, y, color = 'r', x_range='auto', y_range='auto', layers= None,
                              only_scatter=False,only_hist=False,alpha=1,label=''):
    ''' plots synapses in the space and the histograms of their distribution'''
    from matplotlib.ticker import NullFormatter
    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    rect_legend = [left_h, bottom_h, 0.2, 0.2]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx, sharex=axScatter)
    axHisty = plt.axes(rect_histy, sharey=axScatter)
    #axLegend = plt.axes([left_h, bottom_h, 0.2, 0.2])

    # no labels
    #axHistx.xaxis.set_major_formatter(nullfmt)+
    #axHisty.yaxis.set_major_formatter(nullfmt)
    plt.setp(axHistx.get_xticklabels(), visible=False)
    plt.setp(axHisty.get_yticklabels(), visible=False)

    # hide some of the axes
    axHisty.spines['right'].set_visible(False)
    axHisty.spines['top'].set_visible(False)
    axHisty.xaxis.set_ticks_position('bottom')
    axHisty.yaxis.set_ticks_position('left')
    axHistx.spines['right'].set_visible(False)
    axHistx.spines['top'].set_visible(False)
    axHistx.xaxis.set_ticks_position('bottom')
    axHistx.yaxis.set_ticks_position('left')
    axScatter.spines['right'].set_visible(False)
    axScatter.spines['top'].set_visible(False)
    axScatter.xaxis.set_ticks_position('bottom')
    axScatter.yaxis.set_ticks_position('left')


    if not only_hist:
        # the scatter plot:
        axScatter.scatter(x, y, color=color, alpha=alpha,label=label)

    # now determine nice limits by hand:
    binwidth = 50
    if x_range == 'auto' or y_range == 'auto':
        xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
        lim = (int(xymax/binwidth) + 1) * binwidth
	bins = np.arange(-lim, lim + binwidth, binwidth)

    if x_range == 'auto':
        axScatter.set_xlim((-lim, lim))
        x_range = [-lim, lim]
    else:
        axScatter.set_xlim(x_range)
        bins = np.linspace(x_range[0], x_range[1] + binwidth, binwidth)
        

    if y_range == 'auto':
        axScatter.set_ylim((-lim, lim))
    else:
        axScatter.set_ylim((y_range[0], y_range[1]))
        bins = np.linspace(y_range[0], y_range[1] + binwidth, binwidth)
    
    if layers != None:
        for layer in layers[1:]:
            if not only_scatter:
                axHisty.hlines(y=layer[0],xmin=0, xmax=200, linestyles='--')
            axScatter.hlines(y=layer[0], xmin=-600, xmax=600, linestyles='--')

    if not only_scatter:
        axHistx.hist(x, bins=bins, color = color, alpha=alpha)
        axHistx.set_title('x')
        axHisty.hist(y, bins=bins, orientation='horizontal', color = color,alpha=alpha)
        axHisty.set_title('y')

        axHistx.set_xlim(axScatter.get_xlim())
        axHisty.set_ylim(axScatter.get_ylim())
    if not only_hist:
        axScatter.legend()
    return axScatter, axHistx, axHisty

def plot_neuron(neuron_numb, cell_path, params_cell, inh_syn_coords=None, exc_syn_coords=None):
    import sim_cell as temp_sim
    ax = plt.gca()

    all_neurons = hl.find_files(cell_path, ext='json')
    for next_neuron in range(neuron_numb):
        # choose random neuron to plot
        selected = random.choice(all_neurons)
        selected = os.path.join(cell_path, selected)

        with file(selected, 'r') as fid:
            params_selected = json.load(fid)

            seg_coords = temp_sim.init_cell(params_cell, params_selected)
            shift_cell = params_selected['cell_coords']
            seg_coords = temp_sim.shift_cell(params_selected, seg_coords)
            draw_soma(ax, x0=shift_cell[0]+10, x1=shift_cell[0]-10,
                  y0=shift_cell[1]+10, y1=shift_cell[1]-10,
                  color='k')
            keep_segs = ['apic', 'dend', 'soma']
            cleaned_seg_coords = cs.choose_subset_of_segs(seg_coords, keep_segs)
            if exc_syn_coords is not None:
                plot_synapses(exc_syn_coords, marker_size=6, color='y')
            if inh_syn_coords is not None:
                plot_synapses(inh_syn_coords, marker_size=6, color='m')

            graph.plot_neuron(cleaned_seg_coords, autolim=False)  

def draw_soma(ax, x0, x1, y0, y1, color):

    from matplotlib.patches import Ellipse
    width = np.abs(x1-x0)
    ellipse = Ellipse(xy=(np.mean([x0,x1]),np.mean([y0,y1])), width=width, height=width, edgecolor=None, fc=color, lw=2, angle=0)
    ax.add_patch(ellipse)

