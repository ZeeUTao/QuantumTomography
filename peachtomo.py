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
from tomo import *


folder = 'data'
files = sorted((fn for fn in os.listdir(folder+'//') if fn.endswith('.csv')))


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

    ax.view_init(azim=-40,elev= 34)

    bar = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors,alpha=0.9,edgecolor='white', linewidth=0.5)
    #bar = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.9)

    if title and fig:
        ax.set_title(title)

    # x axis
    ax.axes.w_xaxis.set_major_locator(plt.IndexLocator(1, bar_width/2))
    if xlabels:
        ax.set_xticklabels(xlabels)
    ax.tick_params(axis='x', labelsize=10)

    # y axis
    ax.axes.w_yaxis.set_major_locator(plt.IndexLocator(1, bar_width/2))
    if ylabels:
        ax.set_yticklabels(ylabels)
    ax.tick_params(axis='y', labelsize=10)

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




def test_qst_0307():
    opI = np.array([[1, 0], [0, 1]])
    ops = ['I', 'X/2', 'Y/2']

    def get_p1list(filename, ref_filename, correct=True):
        def get_newp1(p1, ref0, ref1):
            p0 = 1 - p1
            newP1 = (ref0 + p1 - 1) / (ref0 + ref1 - 1)
            # newmat = np.dot(newmat, np.linalg.inv(Scale_mat))
            return newP1

        def get_refer(ref_filename):
            ifile = open(folder + '//' + ref_filename, 'r')
            csv_reader = csv.reader(ifile)
            csv_row = np.array([row for row in csv_reader], dtype=float)
            ref0 = csv_row[:, 3]
            ref1 = csv_row[:, 2]
            return np.mean(ref0), np.mean(ref1)

        # the col where p1 of I X/2 Y/2 locate
        col = [2, 4, 6]
        ifile = open(folder + '//' + filename, 'r')
        csv_reader = csv.reader(ifile)
        csv_row = [row for row in csv_reader]
        csv_row = np.array(csv_row, dtype=float)
        data = []
        for l in col:
            data.append(csv_row[:, l])
        data = np.array(data)
        p1list = np.mean(data, 1)
        if correct:
            ref0, ref1 = get_refer(ref_filename)
            p1list = get_newp1(p1list, ref0, ref1)
        return p1list

    def get_densitymat(filename, ref_filename, correct=True):
        # qst filename

        def get_xyz(p1list):
            # ops = ['I','X/2','Y/2']
            z = 1 - 2 * p1list[0]
            y = 1 - 2 * p1list[1]
            x = 2 * p1list[2] - 1
            return x, y, z

        p1list = get_p1list(filename, ref_filename, correct)
        x, y, z = get_xyz(p1list)
        mat = np.array([[1 + z, x - 1j * y], [x + 1j * y, 1 - z]]) / 2.0
        return mat
    filenames = sorted((fn for fn in os.listdir('data//') if fn.endswith('.csv')))[0:5]
    # filenames = sorted((fn for fn in os.listdir('.') if fn.endswith('.csv')))
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
    return




# see tomo
def tomo_mat(filename=files[1],ref_filename=files[0],correct=True):
    def get_refer(ref_filename):
        ifile = open(folder + '//' + ref_filename, 'r')
        csv_reader = csv.reader(ifile)
        csv_row = np.array([row for row in csv_reader], dtype=float)
        ref0 = csv_row[:, 3]
        ref1 = csv_row[:, 2]
        return np.mean(ref0), np.mean(ref1)
    def get_newp1(p1, ref0, ref1):
        p0 = 1 - p1
        newP1 = (ref0 + p1 - 1) / (ref0 + ref1 - 1)
        # newmat = np.dot(newmat, np.linalg.inv(Scale_mat))
        return newP1
    # the col where p1 of I X/2 Y/2 locate
    col =[2,4,6]
    ifile = open(folder+'//'+filename, 'r')
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
    p0list = 1 - p1list
    mat32 = np.array([p0list,p1list]).T
    # format:
    # [[1,0],[1,0],[1,0]]
    return mat32


def get_erho(filename=files[1],ref_filename=files[0],correct=True):
    mat = tomo_mat(filename,ref_filename,correct)
    rho = Qobj(qst(mat, 'tomo')).unit()
    return rho.full()


def test_qpt(filesList=files[1:5],ref_filename=files[0],correct=True):
    """Generate a random chi matrix, and check that we
    recover it from process tomography.
    """
    def get_rhos(Nqubits = 1):
        # create input density matrices from a bunch of rotations
        ops = [np.eye(2), Xpi2, Ypi2, Xpi]
        Us = tensor_combinations(ops, Nqubits)
        rho = np.zeros((2**Nqubits, 2**Nqubits))
        # initial as |0><0| start state U|0><0|U.dag()
        rho[0, 0] = 1
        rhos = [dot3(U, rho, U.conj().T) for U in Us]
        return rhos
    Erhos =[]
    rhos = get_rhos(Nqubits = 1)
    for files in filesList:
        erho = get_erho(files,ref_filename,correct)
        Erhos.append(erho)
    Erhos = np.array(Erhos)
    chi = qpt(rhos, Erhos, T='sigma',return_all=False)
    return chi


def qpt_run0307():
    interval = 5

    fig = plt.figure(figsize=(12,6))
    order = ['a','b','c','d']
    def set_figorder(ax, order='a',x=0.2, y= 0.95):
        ax.set_title(order, x=x, y=y, fontsize=14)
        return

    for idx,i in enumerate([0,2,4,5]):
        step = 5*i
        chi = test_qpt(filesList=files[1+step:5+step],ref_filename=files[0+step])
        op_label = ["I", "X", "Y", "Z"]

        ax = fig.add_subplot(2,4,2*idx+1,projection='3d')
        set_figorder(ax, order=order[idx])
        matrix_histogram_tao(np.real(chi),xlabels=op_label, ylabels=op_label,ax=ax)
        ax.set_zlim3d(-1, 1)


        ax = fig.add_subplot(2,4,2*idx+2,projection='3d')
        matrix_histogram_tao(np.imag(chi), xlabels=op_label, ylabels=op_label,ax=ax)
        ax.set_zlim3d(-1, 1)
        fig.tight_layout()
        #matrix_histogram_tao(chi, xlabels=op_label, ylabels=op_label)
    return

def qpt_run0423():
    interval = 5

    fig = plt.figure(figsize=(8,6))
    order = ['a','b','c','d']
    def set_figorder(ax, order='a',x=0.2, y= 0.95):
        ax.set_title(order, x=x, y=y, fontsize=14)
        return

    for idx,i in enumerate([0]):
        step = 5*i
        chi = test_qpt(filesList=files[1+step:5+step],ref_filename=files[0+step])
        op_label = ["I", "X", "Y", "Z"]
        
        ax = fig.add_subplot(1,1,2*idx+1,projection='3d')
        set_figorder(ax, order=order[idx])
        matrix_histogram_complex((chi),xlabels=op_label, ylabels=op_label,ax=ax)
        
        # ax = fig.add_subplot(1,2,2*idx+1,projection='3d')
        # set_figorder(ax, order=order[idx])
        # matrix_histogram_tao(np.real(chi),xlabels=op_label, ylabels=op_label,ax=ax)
        # ax.set_zlim3d(-1, 1)


        # ax = fig.add_subplot(1,2,2*idx+2,projection='3d')
        # matrix_histogram_tao(np.imag(chi), xlabels=op_label, ylabels=op_label,ax=ax)
        # ax.set_zlim3d(-1, 1)
        # fig.tight_layout()
        #matrix_histogram_tao(chi, xlabels=op_label, ylabels=op_label)
    return chi
    
# qpt_run0307()

# plt.show()


# test_qst_0307()
# plt.show()

