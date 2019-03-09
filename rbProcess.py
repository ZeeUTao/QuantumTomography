import numpy as np
from numpy import sin, cos, sqrt, arctan, pi, exp
import matplotlib.pyplot as plt
import pylab as pl
from scipy import integrate
from scipy.integrate import quad, dblquad, nquad
import math
import scipy
import csv
import os



num_n = 35
num_repeat = 40

def get_refer(filename ='00391 - %vq2%g%c Randomized Benchmarking 1q reference.csv'):
    ifile = open(filename, 'r')
    reader = csv.reader(ifile)
    n = []
    p0 = []
    for idx, row in enumerate(reader):
        n.append(float(row[1]))
        p0.append(float(row[2]))
    n, p0 = np.array(n), np.array(p0)
    n, p0 = n.reshape((num_repeat, num_n)), p0.reshape((num_repeat, num_n))
    return n.T, p0.T





def plot_refer_raw():
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    for idx,n in enumerate(nlist):
        ax.scatter(n, p0[idx], s = 2 ,alpha = 0.5)
    ax.plot(nlist[:,0],np.mean(p0,1),linewidth = 2,c='r', marker='o')
    return




def fit_sequence(nlist, p0,color = 'k',label ='ref',extract_ref = None):
    # fit the average sequence fidelity
    n0_refer = nlist[:,0]
    p0_refer = np.mean(p0,1)
    p0_refer_std = np.std(p0,1)


    def fitfunc(p, m):
        return p[0] * np.power(p[1],m) + p[2]

    def errfunc(p):
        return p0_refer - fitfunc(p, n0_refer)

    out = scipy.optimize.leastsq(errfunc, np.array([p0_refer[0],0.99,p0_refer[-1]]), full_output=1)

    para = out[0]

    #p0_err = np.std(np.power(p0 / para[0], 1 / nlist), 1)
    p0_err = np.power(p0_refer_std,n0_refer)


    if extract_ref == None:
        fid = para[1]
        r_clii = (1.0- fid) * 0.5
        r_ave = r_clii / 1.875
        show_fid = 1.0-r_clii

    else:
        r_gate = (1 -para[1] /extract_ref  ) * 0.5
        fid = 1-r_gate
        show_fid =fid



    p0_refer_fit = fitfunc(p = [0.5,para[1],0.48],m=n0_refer)

    p0_data_renorm = (p0_refer-para[2])/para[0] *0.5 +0.48
    # plt.scatter(n0_refer,(p0_refer-para[2])/para[0] *0.5 +0.5,c=color,s=30
    #             ,label =label + ':'+str(round(fid,4)) + '('+')')
    eb = plt.errorbar(n0_refer, p0_data_renorm, yerr=p0_refer_std,
                      c='none',ms=5,ecolor=color,elinewidth=2,capsize=4,alpha = 1)
    plt.scatter(n0_refer, p0_data_renorm,c=color )

    show_label = str(round(show_fid,4)) + '('+str(np.int(1e4 *np.mean(p0_err) ))+')'

    plt.plot(n0_refer,p0_refer_fit,c=color
                ,label =label )#+show_label)


    plt.ylim(0.45,1.0)
    plt.legend()
    return fid,show_label

# plt.scatter(n0_refer,p0_refer)

def run_rb(filename,color = 'k',label ='refer',extract_ref = None):
    nlist, p0 = get_refer(filename =filename)
    res = fit_sequence(nlist, p0,color = color,label =label,extract_ref=extract_ref)
    return res


plt.figure(figsize=(8,6))
filenames = sorted((fn for fn in os.listdir('.') if fn.endswith('.csv')))
label = [r'$p_{ref}$','X','X/2','Y','Y/2','-X','-X/2','Y','-Y/2']

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