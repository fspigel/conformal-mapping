import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pflow
from progressbar import progressbar as pb

window = tk.Tk()
window.title('FMPLT')

subframe = tk.Frame(master=window)
subframe.grid(row=0, column=0, sticky=tk.W)
subframe2 = tk.Frame(master=window)
subframe2.grid(row=1, column=0, sticky=tk.W)

label_0 = tk.Label(master=subframe, text='PHI: ')
label_0.grid(row=0, column=0, sticky=tk.W)
label_1 = tk.Label(master=subframe, text='PSI: ')
label_1.grid(row=1, column=0)

# axis settings
lbl_x_axis = tk.Label(master=subframe, text='X axis: ')
lbl_x_axis.grid(row=3, column=0, sticky=tk.W)
txt_x_axis_from = tk.Entry(master=subframe, width=10)
txt_x_axis_from.insert(tk.END, '-2')
txt_x_axis_from.grid(row=3, column=1, sticky=tk.W)
txt_x_axis_to = tk.Entry(master=subframe, width=10)
txt_x_axis_to.insert(tk.END, '2')
txt_x_axis_to.grid(row=3, column=2, sticky=tk.W)
lbl_y_axis = tk.Label(master=subframe, text='Y axis: ')
lbl_y_axis.grid(row=4, column=0, sticky=tk.W)
txt_y_axis_from = tk.Entry(master=subframe, width=10)
txt_y_axis_from.insert(tk.END, '-2')
txt_y_axis_from.grid(row=4, column=1, sticky=tk.W)
txt_y_axis_to = tk.Entry(master=subframe, width=10)
txt_y_axis_to.insert(tk.END, '2')
txt_y_axis_to.grid(row=4, column=2, sticky=tk.W)


txt_0 = tk.Entry(master=subframe, width=35, )
txt_0.insert(tk.END, 'x*(1+1/(x**2+y**2))')
txt_0.grid(row=0, column=1, sticky=tk.W, columnspan=2)
txt_1 = tk.Entry(master=subframe, width=35)
txt_1.insert(tk.END, 'y*(1-1/(x**2+y**2))')
txt_1.grid(row=1, column=1, sticky=tk.W, columnspan=2)


main_tab = ttk.Notebook(subframe2)
tab_vec = ttk.Frame(main_tab)
tab_str = ttk.Frame(main_tab)
main_tab.add(tab_vec, text = 'Vector view')
main_tab.add(tab_str, text = 'Streamline view')
main_tab.pack(expand=1, fill='both')

xlim_global = [-2,2]
ylim_global = [-2,2]

fig_vec = plt.Figure()
ax_vec = fig_vec.add_subplot(111)
ax_vec.plot([], [])
ax_vec.set_xlim(xlim_global)
ax_vec.set_ylim(ylim_global)
canvas_vec = FigureCanvasTkAgg(fig_vec, master=tab_vec)
canvas_vec.draw()
canvas_vec.get_tk_widget().grid(row=4, column=0, sticky=tk.W, columnspan=3)

fig_str = plt.Figure()
ax_str = fig_str.add_subplot(111)
ax_vec.set_xlim(xlim_global)
ax_vec.set_ylim(ylim_global)
ax_str.plot([], [])
ax_str.set_xlim(xlim_global)
ax_str.set_ylim(ylim_global)
canvas_str = FigureCanvasTkAgg(fig_str, master=tab_str)
canvas_str.draw()
canvas_str.get_tk_widget().grid(row=4, column=0, sticky=tk.W, columnspan=3)

line_hor = None
line_ver = None
pflow_module = None
single_streamline = None
streamlines = None
particle_plots = None
point_plot = None
animation_grid = None
animation_plots = None
movement_interval = 0.1
particle_iterator = 0
particle_period = 5


def plot():
    global xlim_global, ylim_global, pflow_module
    xlim_global = [float(txt_x_axis_from.get()), float(txt_x_axis_to.get())]
    ylim_global = [float(txt_y_axis_from.get()), float(txt_y_axis_to.get())]

    ax_vec.clear()
    ax_vec.set_xlim(xlim_global)
    ax_vec.set_ylim(ylim_global)

    ax_str.clear()
    ax_str.set_xlim(xlim_global)
    ax_str.set_ylim(ylim_global)

    phi_str = txt_0.get()
    psi_str = txt_1.get()
    pflow_module = pflow.Pflow.from_txt(phi_str, psi_str)
    plot_vector_field()
    ax_vec.scatter([0.512], [0.918], marker='x')
    canvas_vec.draw()
    plot_streamlines()
    generate_point_animation_grid()
    generate_animation_plots()
    move_particles()
    canvas_vec.draw()
    # move_single_point([-2, 0.5], streamlines[int(len(streamlines)/2)])

# plot button
plot_button = tk.Button(master=subframe, height=1, width=6, text='Plot', command=plot)
plot_button.grid(row=0, column=3, rowspan=2, sticky='nsew')

def plot_vector_field():
    # fig_vec.clear()
    n_x, n_y = 40, 40
    pflow_module.get_vector_field(xlim_global, ylim_global, n_x, n_y)
    R, V = pflow_module.get_plottable()
    X = np.linspace(xlim_global[0], xlim_global[1], n_x)
    Y = np.linspace(ylim_global[0], ylim_global[1], n_y)
    X_msh, Y_msh = np.meshgrid(X, Y)
    default = [ax_vec.get_xlim()[0]-1, ax_vec.get_ylim()[0]-1]
    for j in range(n_x):
        for i in range(n_y):
            if np.linalg.norm([X_msh[i, j], Y_msh[i, j]]) < 1:
                X_msh[i, j] = default[0]
                Y_msh[i, j] = default[1]
    ax_vec.quiver(X_msh, Y_msh, V[:,:,0], V[:,:,1], R, alpha=.5)
    # ax_vec.plot([0,1], [0,1])
    canvas_vec.draw()

def plot_streamlines():
    global streamlines
    alpha = np.linspace(0, 2 * np.pi, 100)
    a = xlim_global[1] - xlim_global[0]
    b = ylim_global[1] - ylim_global[0]
    r = 1.1 * np.sqrt((a ** 2 + b ** 2)) / 2
    x_circle = r * np.cos(alpha)
    y_circle = r * np.sin(alpha)

    streamlines = []
    for point in zip(x_circle, y_circle):
        try:
            streamline = pflow_module.get_streamline(point, xlim_global)
            streamlines.append(streamline)
            ax_str.plot(streamline[0,:], streamline[1,:], 'C0')
        except ValueError as e:
            print(e)
            continue
    ax_str.set_xlim(xlim_global)
    ax_str.set_ylim(ylim_global)
    canvas_str.draw()

def generate_animation_plots():
    print('generating animation plots', flush=True)
    global animation_plots
    animation_plots = []
    for i in pb(range(particle_period)):
        animation_plot_x = []
        animation_plot_y = []
        for animation_line in animation_grid:
            animation_plot_x = np.append(animation_plot_x, animation_line[0,i::particle_period])
            animation_plot_y = np.append(animation_plot_y, animation_line[1,i::particle_period])
        animation_plots.append([animation_plot_x, animation_plot_y])

def move_particles():
    # print('move')
    global particle_iterator, particle_plots
    if particle_plots != None:
        for particle_plot in particle_plots:
            particle_plot.remove()
    particle_plots = []
    for streamline_animation in animation_grid:
        # particles_y = streamline[1][particle_iterator::particle_period]
        # ax_str.scatter(animation_plots[particle_iterator][0],
        #                                 animation_plots[particle_iterator][1],
        #                                 c='C2')
        particle_plot = ax_str.scatter(animation_plots[particle_iterator][0],
                                        animation_plots[particle_iterator][1],
                                        c='C2')
        particle_plots.append(particle_plot)
    particle_iterator = particle_iterator+1
    if particle_iterator == particle_period:
        particle_iterator = 0
    canvas_str.draw()
    # print('done')
    window.after(50, move_particles)

def get_velocity(point):
    points = []
    for i in range(len(pflow_module.X)):
        for j in range(len(pflow_module.Y)):
            points.append({'position' : np.array([pflow_module.X[i], pflow_module.Y[j]]),
                           'distance' : np.linalg.norm(np.array([pflow_module.X[i], pflow_module.Y[j]])-point),
                           'i' : i,
                           'j' : j})
    points.sort(key = lambda x: x['distance'])
    points = points[:4]
    v = [np.linalg.norm(pflow_module.U[x['i'],x['j']]) for x in points]
    return sum(v)/4
    # close_X = min(pflow_module.X, key=lambda x:abs(x-point[0]))
    # i, = np.where(np.isclose(pflow_module.X, close_X))
    # close_Y = min(pflow_module.Y, key=lambda x:abs(x-point[1]))
    # j, = np.where(np.isclose(pflow_module.Y, close_Y))
    # return np.linalg.norm(pflow_module.U[j,i])

# test function, temporary
def move_single_point(point, streamline):
    global point_plot
    new_position = move_point(point, streamline)
    if point_plot != None:
        point_plot.remove()
    point_plot = ax_str.scatter([new_position[0]], [new_position[1]], c='C2')
    canvas_str.draw()
    window.after(50, move_single_point(new_position, streamline))

def generate_point_animation_grid():
    global animation_grid
    animation_grid = []
    i = 0
    print('generating animation grid', flush=True)
    for streamline in pb(streamlines):
        point = streamline[:,0]
        i = i + 1
        streamline_animation = [point]
        flag = False
        j = 0
        while not flag:
            j = j + 1
            new_point, flag = move_point(streamline_animation[-1], streamline)
            streamline_animation.append(new_point)
        streamline_animation.pop() # last element is garbage
        streamline_animation = np.array(streamline_animation).transpose()
        animation_grid.append(streamline_animation)
    return animation_grid

def animate_points(point_set, streamline_index):
    new_point_set = []
    x_values = []
    y_values = []
    for i, point in enumerate(point_set):
        new_position = move_point(point, streamlines[streamline_index])
        new_point_set.append(new_position)
        x_values.append(new_position[0])
        y_values.append(new_position[1])
    ax_str.scatter(x_values, y_values, c='C2')
    canvas_str.draw()

def move_point(point, streamline):
    v = get_velocity(point) # current velocity
    dis_remaining = v*movement_interval # distance to travel
    h = streamline[0][1] - streamline[0][0]
    index_str = int(np.floor((point[0]-streamline[0,0])/h))     # index of the closest streamline point on the left
                                                                # side of the given point
    next_str = streamline[:,index_str+1]
    # next_str = np.array([streamline[0,index_str+1], streamline[1,index_str+1]])

    dis_next = np.linalg.norm(point-next_str)
    if dis_remaining < dis_next:
        rem = dis_remaining/dis_next
        new_position = point + rem*(next_str-point)
        # print('current: {} next: {} ')
        return new_position, False

    while dis_remaining > dis_next:
        # print('rem: {:.5f}; next: {:.5f}'.format(dis_remaining, dis_next))
        # print(index_str)
        if index_str+2 >= streamline.shape[1]:
            return streamline[:,-1], True
            # return np.array([streamline[0,-1], streamline[1,-1]]), True
        dis_remaining = dis_remaining - dis_next
        index_str = index_str + 1
        dis_next, current_str, dir_next = get_streamline_point_distance(streamline, index_str)

    rem = dis_remaining/dis_next
    new_position = current_str + rem*(dir_next)
    return new_position, False

    # if d1 > d0:
    #     rem = d0/d1
    #     new_position = point + rem*(next_str-point)
    #     return new_position
    # d0 = d0-d1
    # d_next = 0
    # while d0 > d_next:
    #     # popraviti, ruzno ruzno ruzno ruzno
    #     next_next_str = np.array([streamline[0][index_str+2], streamline[1][index_str+2]])
    #     d_next = np.linalg.norm(next_str - next_next_str)
    #     d0 = d0-d_next
    #     index_str = index_str+1
    #     next_str = next_next_str

def get_streamline_point_distance(streamline, index):
    current = np.array([streamline[0][index], streamline[1][index]])
    next = np.array([streamline[0][index+1], streamline[1][index+1]])
    return np.linalg.norm(next-current), current, next-current

def motion(event):
    global line_hor, line_ver, single_streamline
    x, y = event.x, event.y
    wx_local = tab_str.winfo_x()
    wy_local = tab_str.winfo_y()
    graph_x_ratio, graph_y_ratio, graph_x_actual, graph_y_actual = convert_coordinates([x, y])
    window.title('FMPLT - mouse: ({}, {}); plot: ({:.2f}, {:.2f}); graph: ({:.2f}, {:.2f})'.format(
        x, y, graph_x_ratio, graph_y_ratio, graph_x_actual, graph_y_actual
    ))

    if line_hor != None:
        line_hor.remove()
    if line_ver != None:
        line_ver.remove()
    # line_hor, = ax_str.plot([graph_x_actual, graph_x_actual], ax_str.get_ylim(), 'black')
    # line_ver, = ax_str.plot(ax_str.get_xlim(), [graph_y_actual, graph_y_actual], 'black')
    if single_streamline != None:
        single_streamline_backup = single_streamline
        try:
            single_streamline.remove()
        except ValueError as e:
            print(e)
    if pflow_module != None:
        try:
            x_str, y_str = pflow_module.get_streamline([graph_x_actual, graph_y_actual], xlim_global)
            single_streamline, = ax_str.plot(x_str, y_str, 'C1')
        except (ValueError, RuntimeError) as e:
            print(e)
            single_streamline = single_streamline_backup
    ax_str.set_xlim(xlim_global)
    ax_str.set_ylim(ylim_global)
    canvas_str.draw()
    #print(x, y, wx, wx_local)
    #print(f'graph: {}')

def convert_coordinates(mouse_coordinates):
    mouse_x, mouse_y = mouse_coordinates[0], mouse_coordinates[1]
    #account for margin
    graph_x = mouse_x - 81
    graph_y = mouse_y - 58
    #convert to percentage
    graph_x_ratio = graph_x/(576-81)
    graph_y_ratio = 1 - graph_y/(428-58) # invert y coordinate
    xlim = ax_vec.get_xlim()
    ylim = ax_vec.get_ylim()
    #convert to graph coordinates
    graph_x_actual = graph_x_ratio*(xlim[1]-xlim[0]) + xlim[0]
    graph_y_actual = graph_y_ratio*(ylim[1]-ylim[0]) + ylim[0]
    return graph_x_ratio, graph_y_ratio, graph_x_actual, graph_y_actual

window.bind('<Motion>', motion)
window.mainloop()