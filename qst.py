import numpy as np
from numpy import sin, cos, sqrt, arctan, pi, exp
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


import csv
import os
from qutip import *

def matrix_histogram_tao(M, xlabels=None, ylabels=None, title=None, limits=None,
                     colorbar=False, fig=None, ax=None):
    """
    Draw a histogram for the matrix M, with the given x and y labels and title.

    Parameters
    ----------
    M : Matrix of Qobj
        The matrix to visualize

    xlabels : list of strings
        list of x labels

    ylabels : list of strings
        list of y labels

    title : string
        title of the plot (optional)

    limits : list/array with two float numbers
        The z-axis limits [min, max] (optional)

    ax : a matplotlib axes instance
        The axes context in which the plot will be drawn.

    Returns
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.

    Raises
    ------
    ValueError
        Input argument is not valid.

    """

    if isinstance(M, Qobj):
        # extract matrix data from Qobj
        M = M.full()

    n = np.size(M)

    bar_width = 0.6
    xpos, ypos = np.meshgrid(np.arange(1,M.shape[0]+1,1), np.arange(1,M.shape[1]+1,1))
    xpos = xpos.T.flatten() - bar_width/2
    ypos = ypos.T.flatten() - bar_width/2
    zpos = np.zeros(n)
    dx = dy = bar_width * np.ones(n)
    dz = np.real(M.flatten())

    if limits and type(limits) is list and len(limits) == 2:
        z_min = limits[0]
        z_max = limits[1]
    else:
        z_min = min(dz)
        z_max = max(dz)
        if z_min == z_max:
            z_min -= 0.1
            z_max += 0.1
    norm = mpl.colors.Normalize(-1, 1)
    #norm = mpl.colors.Normalize(z_min, z_max)
    # cmap = cm.get_cmap('jet')  # Spectral
    # cmap = plt.get_cmap('bwr')
    #cmap = plt.get_cmap('GnBu') RdBu
    cmap = plt.get_cmap('coolwarm')
    colors = cmap(norm(dz),alpha=0.9)

    if ax is None:
        fig = plt.figure(figsize=(10,8))
        ax = Axes3D(fig, azim=-32, elev=74)

    ax.view_init(azim=-32,elev= 16)

    bar = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors,alpha=0.9,edgecolor='white', linewidth=0.5)
    #bar = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.9)

    if title and fig:
        ax.set_title(title)

    # x axis
    ax.axes.w_xaxis.set_major_locator(plt.IndexLocator(1, bar_width/2))
    if xlabels:
        ax.set_xticklabels(xlabels)
    ax.tick_params(axis='x', labelsize=14)

    # y axis
    ax.axes.w_yaxis.set_major_locator(plt.IndexLocator(1, bar_width/2))
    if ylabels:
        ax.set_yticklabels(ylabels)
    ax.tick_params(axis='y', labelsize=14)

    # z axis
    ax.axes.w_zaxis.set_major_locator(plt.IndexLocator(1, 0.5))
    ax.set_zlim3d([min(z_min, 0), z_max])
    # ax.set_xticks(np.arange(1,max_num+2,2))
    # ax.set_yticks(np.arange(1, max_num+2, 2))
    # color axis
    if colorbar:
        #cbar = fig.colorbar(bar, ax=ax, shrink=0.5, aspect=10)
        cax, kw = mpl.colorbar.make_axes(ax, shrink=1.0, pad=-0.0)
        mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)

    # ax.grid(color='r', linestyle='', linewidth=0)
    return fig, ax



opI = np.array([[1,0],[0,1]])
ops = ['I','X/2','Y/2']


def get_newp1(p1,ref0, ref1):
    p0 = 1 - p1
    newP1 = (ref0 + p1 -1) / (ref0 + ref1 - 1)
    # newmat = np.dot(newmat, np.linalg.inv(Scale_mat))
    return newP1



def get_densitymat(filename,ref_filename,correct=True):
    # qst filename
    def get_refer(ref_filename):
        ifile = open(ref_filename, 'r')
        reader = csv.reader(ifile)
        ref0 = []
        ref1 = []
        for idx, row in enumerate(reader):
            ref0.append(float(row[3]))
            ref1.append(float(row[2]))
        ref0 = np.array(ref0)
        ref1 = np.array(ref1)
        return np.mean(ref0), np.mean(ref1)

    def get_p1list(filename):
        # the col where p1 of I X/2 Y/2 locate
        col =[2,4,6]
        ifile = open(filename, 'r')
        csv_reader = csv.reader(ifile)
        csv_row = [row for row in csv_reader]
        csv_row = np.array(csv_row, dtype=float)
        data = []
        for l in col:
            data.append(csv_row[:, l])
        data = np.array(data)
        p1list = np.mean(data,1)
        if correct:
            ref0, ref1 = get_refer(ref_filename)
            p1list = get_newp1(p1list,ref0, ref1)
        return p1list
    def get_xyz(p1list):
        # ops = ['I','X/2','Y/2']
        z = 1 - 2 * p1list[0]
        y = 1 - 2 * p1list[1]
        x = 2 * p1list[2] - 1
        return x, y, z


    p1list = get_p1list(filename)
    x, y, z = get_xyz(p1list)
    mat = np.array([[1+z,x-1j*y],[x+1j*y,1-z]]) /2.0
    return mat












filenames = sorted((fn for fn in os.listdir('.') if fn.endswith('.csv')))
label = [r'ref','I','X/2','X','-X','Y/2','Y/2']

ref_filename = filenames[0]
matlist_new = []
matlist_raw = []
for i in range(len(filenames)-1):
    filename = filenames[i+1]
    mat_new = get_densitymat(filename,ref_filename,correct=True)
    mat_raw = get_densitymat(filename,ref_filename,correct=False)
    matlist_raw.append(mat_raw)
    matlist_new.append(mat_new)


def ideal_rho(gate):
    psi = gate * basis(2,0)
    return psi*psi.dag()
g = basis(2,0)
f = basis(2,1)

xlabels=[r'0',r'1']
ylabels=[r'0',r'1']
def set_figorder(ax, order='a',x=0.2, y= 0.95):
    ax.set_title(order, x=x, y=y, fontsize=18)
    return

order = ['a','b','c','d']
fig = plt.figure(figsize=(12,6))
gate = [qeye(2),rx(np.pi/2),ry(np.pi/2),rx(np.pi)]
for i in range(4):
    rho_exp=Qobj(matlist_new[i]).unit()
    ax = fig.add_subplot(2, 4, 2 * i + 1,projection='3d')
    set_figorder(ax, order=order[i])
    matrix_histogram_tao(Qobj(np.real(rho_exp.full())),ax=ax,xlabels=xlabels,ylabels=ylabels)
    ax.set_zlim3d(-1,1)
    ax2 = fig.add_subplot(2, 4, 2 * i + 2,projection='3d')
    matrix_histogram_tao(Qobj(np.imag(rho_exp.full())),ax=ax2,xlabels=xlabels,ylabels=ylabels)
    ax2.set_zlim3d(-1, 1)

fig.tight_layout()


# fig = plt.figure(figsize=(12,6))
# gate = [qeye(2),rx(np.pi/2),ry(np.pi/2),rx(np.pi)]
# for i in range(4):
#     rho_exp=ideal_rho(gate[i])
#     ax = fig.add_subplot(2, 4, 2 * i + 1,projection='3d')
#     matrix_histogram_tao(Qobj(np.real(rho_exp.full())),ax=ax,xlabels=xlabels,ylabels=ylabels)
#     ax.set_zlim3d(-1,1)
#     ax2 = fig.add_subplot(2, 4, 2 * i + 2,projection='3d')
#     matrix_histogram_tao(Qobj(np.imag(rho_exp.full())),ax=ax2,xlabels=xlabels,ylabels=ylabels)
#     ax2.set_zlim3d(-1, 1)

fig.tight_layout()
plt.show()



'''
color= plt.cm.hsv(np.linspace(0,1,len(label)-1))
#gist_rainbow jet
# color = ['k'] + [None]*8

fid_ref = 1

show_label_list = []
for idx,filename in enumerate(filenames):
    # the location of ref .csv in the order of datalist(filenames)
    if idx == 0:
        fid_ref,show_label = run_rb(filenames[idx],color ='k',label =label[idx])
        print('p',fid_ref)
        show_label_list.append(show_label)
    else:
        fid,show_label = run_rb(filenames[idx], color=color[idx-1], label=label[idx],extract_ref = fid_ref)
        show_label_list.append(show_label)

r_clii = (1-fid_ref) * 0.5
r_ave = r_clii / 1.875

print('p_clii',1-r_clii)
print('p_gate',1-r_ave)

for i in show_label_list:
    print(i)

np.save('show_label_list',show_label_list)
fontsize= 14
plt.xlabel('Number of Cliffords',fontsize=fontsize)
plt.ylabel('Sequence Fidelity',fontsize=fontsize)
plt.xlim(-0.1,250)
# plt.grid(alpha = 0.5)


plt.show()
'''
