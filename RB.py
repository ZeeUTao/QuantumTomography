import numpy as np
import numpy.random
import labrad
import pyle.envelopes as env
import matplotlib.pyplot as plt
from pyle.util import sweeptools as st
# import generateSeq_gate_full
# import generateSeq_cz
# import generateSeq_gate
from pyle.pipeline import returnValue, FutureList
#from pyle.dataking import envelopehelpers_qguo as eh
from pyle.dataking import envelopehelpers as eh
from pyle.math import tensor
from pyle import tomo as tomo
from pyle.dataking import sweeps
from pyle.dataking import util
from pyle.dataking.fpgaseq_qnd import runQubits as runQ
import sys
import multiplexed as mp
from pyle.dataking.optimization import optimization as popt
import labrad
import os
import time
from pyle.util import convertUnits
import numpy.random
import scipy
import scipy.io
from labrad.units import Unit,Value
V, mV, us, ns, GHz, MHz, dBm, rad = [Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'MHz', 'dBm', 'rad')]

I2 = np.diag([1,1])
sigmax = np.array([[0,1],[1,0]])
sigmay = np.array([[0,-1j],[1j,0]])
sigmaz = np.array([[1,0],[0,-1]])
XZ = np.array([[1,1],[1,-1]])/np.sqrt(2)
X = -1j*sigmax
Y = -1j*sigmay
Z = -1j*sigmaz
mX = 1j*sigmax
mY = 1j*sigmay
mZ = 1j*sigmaz
halfX = (I2+X)/2**0.5
halfY = (I2+Y)/2**0.5
halfZ = (I2+Z)/2**0.5
mhalfX = (I2-X)/2**0.5
mhalfY = (I2-Y)/2**0.5
mhalfZ = (I2-Z)/2**0.5
iSwapGate = np.array([[1, 0, 0, 0],[0, 0, 1j, 0],[0, 1j, 0, 0],[0, 0, 0, 1]])
swapGate = np.array([[1, 0, 0, 0],[0, 0, 1, 0],[0, 1, 0, 0],[0, 0, 0, 1]])
cNotGate = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0]])
czGate = np.diag([1, 1, 1 , -1])    #Create the matrix for the rotation
S1 = np.array([0,8,7])
S1X = np.array([12,22,15])
S1Y = np.array([14,20,17])
#============constructing Clifford Group====================
def CliffordFor1(q, start, idx, cosEnv=False):
    '''
    Clifford group for one qubit.
    Parameters:
    q: the target qubit 
    start: the start point of the sequence  
    idx(integer): the number for the Clifford.
    cosEnv: |----if False, use Gaussian Envelope, regikey.(piFWHM, piAmp, piDf) for pi pulse, and regikey.(piFWHMHalf, piAmpHalf, piDf) for half pi pulse
            |----if True, use Cosine Envelope,  regikey.(piLen, piAmpCosine, piDf) for pi pulse, and regikey.(piLenHalf, piAmpHalfCosine, piDf) for half pi pulse
    return sequence, end time of the sequence, and the operator matrix of the Clifford
    '''
    operator=[]
    if idx == 0:
        #-------pseudo-identity operation-----------
        # xy = eh.mix(q,eh.piPulse(q,start,phase=0))
        # start += q.piLen
        # xy += eh.mix(q,eh.piPulse(q,start,phase=np.pi))
        #------------------
        op = I2
        operator.append('I')
    elif idx == 1:
        op = X
        operator.append('X')
    elif idx == 2:
        op = Y
        operator.append('Y')
    elif idx == 3:
        op = np.dot(X,Y)
        operator.append('Y')
        operator.append('X')
    elif idx == 4:
        op = np.dot(halfY, halfX)
        operator.append('X/2')
        operator.append('Y/2')
    elif idx == 5:
        op = np.dot(mhalfY, halfX)
        operator.append('X/2')
        operator.append('-Y/2')
    elif idx == 6:
        op = np.dot(halfY, mhalfX)
        operator.append('-X/2')
        operator.append('Y/2')
    elif idx == 7:
        op = np.dot(mhalfY, mhalfX)
        operator.append('-X/2')
        operator.append('-Y/2')
    elif idx == 8:
        op = np.dot(halfX, halfY)
        operator.append('Y/2')
        operator.append('X/2')
    elif idx == 9:
        op = np.dot(mhalfX, halfY)
        operator.append('Y/2')
        operator.append('-X/2')
    elif idx == 10:
        op = np.dot(halfX, mhalfY)
        operator.append('-Y/2')
        operator.append('X/2')
    elif idx == 11:
        op = np.dot(mhalfX, mhalfY)
        operator.append('-Y/2')
        operator.append('-X/2')
    elif idx == 12:
        op = halfX
        operator.append('X/2')
    elif idx == 13:
        op = mhalfX
        operator.append('-X/2')
    elif idx == 14:
        op = halfY
        operator.append('Y/2')
    elif idx == 15:
        op = mhalfY
        operator.append('-Y/2')
    elif idx == 16:
        op = mhalfX
        op = np.dot(halfY,op)
        op = np.dot(halfX,op)
        operator.append('-X/2')
        operator.append('Y/2')
        operator.append('X/2')
    elif idx == 17:
        op = mhalfX
        op = np.dot(mhalfY,op)
        op = np.dot(halfX,op)
        operator.append('-X/2')
        operator.append('-Y/2')
        operator.append('X/2')
    elif idx == 18:
        op = X
        op = np.dot(halfY,op)
        operator.append('X')
        operator.append('Y/2')
    elif idx == 19:
        op = X
        op = np.dot(mhalfY,op)
        operator.append('X')
        operator.append('-Y/2')
    elif idx == 20:
        op = Y
        op = np.dot(halfX,op)
        operator.append('Y')
        operator.append('X/2')
    elif idx == 21:
        op = Y
        op = np.dot(mhalfX,op)
        operator.append('Y')
        operator.append('-X/2')
    elif idx == 22:
        op = halfX
        op = np.dot(halfY,op)
        op = np.dot(halfX,op)
        operator.append('X/2')
        operator.append('Y/2')
        operator.append('X/2')
    elif idx == 23:
        op = mhalfX
        op = np.dot(halfY,op)
        op = np.dot(mhalfX,op)
        operator.append('-X/2')
        operator.append('Y/2')
        operator.append('-X/2')
    else: 
        raise Exception('unrecognized Cliffords number.')
    xy = env.NOTHING
    for idx_seq in range(len(operator)):
        start += q.piLen/2.0
        if cosEnv:
            if operator[idx_seq] == 'I':
                xy += env.NOTHING
            elif operator[idx_seq] == 'X':
                xy += eh.mix(q,eh.piPulseCosine(q,start,phase=0))   
            elif operator[idx_seq] == 'X/2':
                xy += eh.mix(q,eh.piHalfPulseCosine(q,start,phase=0))
            elif operator[idx_seq] == '-X/2':
                xy += eh.mix(q,eh.piHalfPulseCosine(q,start,phase=np.pi)) 
            elif operator[idx_seq] == 'Y':
                xy += eh.mix(q,eh.piPulseCosine(q,start,phase=np.pi/2.0)) 
            elif operator[idx_seq] == 'Y/2':
                xy += eh.mix(q,eh.piHalfPulseCosine(q,start,phase=np.pi/2.0))
            elif operator[idx_seq] == '-Y/2':
                xy += eh.mix(q,eh.piHalfPulseCosine(q,start,phase=-np.pi/2.0))
            else:
                raise Exception('No gate operation for such signal!')
        else:
            if operator[idx_seq] == 'I':
                xy += env.NOTHING
            elif operator[idx_seq] == 'X':
                xy += eh.mix(q,eh.piPulse(q,start,phase=0))   
            elif operator[idx_seq] == 'X/2':
                xy += eh.mix(q,eh.piHalfPulse(q,start,phase=0))
            elif operator[idx_seq] == '-X/2':
                xy += eh.mix(q,eh.piHalfPulse(q,start,phase=np.pi)) 
            elif operator[idx_seq] == 'Y':
                xy += eh.mix(q,eh.piPulse(q,start,phase=np.pi/2.0)) 
            elif operator[idx_seq] == 'Y/2':
                xy += eh.mix(q,eh.piHalfPulse(q,start,phase=np.pi/2.0))
            elif operator[idx_seq] == '-Y/2':
                xy += eh.mix(q,eh.piHalfPulse(q,start,phase=-np.pi/2.0))
            else:
                raise Exception('No gate operation for such signal!')
        start += q.piLen/2.0
    
    return start, xy, op
          
    
def CliffordForN(qubits, start, idxs):
    '''
        Pauli-based Clifford Groups for N qubit gate.
        idxs record Clifford gate for each qubit in qubits
        return end time of the sequence, sequence, and the operator matrix of the Clifford
    '''
    qubitNum = len(qubits)
    xys = [env.NOTHING]*qubitNum
    start0 = start
    starts, ops = [],[]
    for idx,qubit in enumerate(qubits):
        start, C, op = CliffordFor1(qubit, start0, idxs[idx])
        xys[idx] += C
        starts.append(start)
        ops.append(op)
    
    start = np.max(starts)
    op = reduce(np.kron, ops)

    return start, xys, op

def fullCliffordFor2(r, qubits, start, idxs, Omega2, delay, delta):
    '''
        full set of Clifford Groups for 2 qubit gate.
        all gate constructed by CZ and single qubit rotation
        idxs is a 2*4 array , indicating which clifford gate should be used:
        q1: idx1-C-idx2-C-idx3-C-idx4
        q2: idx1-Z-idx2-Z-idx3-Z-idx4
        return end time of the sequence, sequence, and the operator matrix of the Clifford
    '''
    
    xys = [env.NOTHING]*3
    zs = [env.NOTHING]*3
    start0 = start
    
    #--------Cs---------
    starts, ops = [],[]
    for idx,qubit in enumerate(qubits):
        start, C, opC = CliffordFor1(qubit, start0, idxs[0,idx])
        xys[idx+1] += C
        starts.append(start)
        ops.append(opC)
    start0 = np.max(starts)
    op = reduce(np.kron, ops)
    #--------CZ---------
    if idxs[1,1] == 1:
        start0, CZxy, CZz, opCZ = geomCZGate(r, qubits, start0, Omega2, delay, delta)
        for idx in range(3):
            xys[idx] += CZxy[idx]
            zs[idx] += CZz[idx]
        op = np.dot(opCZ, op)
        #--------Ss---------
        starts, ops = [],[]
        for idx,qubit in enumerate(qubits):
            start, C, opC = CliffordFor1(qubit, start0, idxs[2,idx])
            xys[idx+1] += C
            starts.append(start)
            ops.append(opC)
        start0 = np.max(starts)
        op = np.dot(reduce(np.kron, ops), op)
        #--------CZ---------
        if idxs[3,1] == 1:
            start0, CZxy, CZz, opCZ = geomCZGate(r, qubits, start0, Omega2, delay, delta)
            for idx in range(3):
                xys[idx] += CZxy[idx]
                zs[idx] += CZz[idx]
            op = np.dot(opCZ, op)
            #--------Ss---------
            starts, ops = [],[]
            for idx,qubit in enumerate(qubits):
                start, C, opC = CliffordFor1(qubit, start0, idxs[4,idx])
                xys[idx+1] += C
                starts.append(start)
                ops.append(opC)
            start0 = np.max(starts)
            op = np.dot(reduce(np.kron, ops), op)
            #--------CZ---------
            if idxs[5,1] == 1:
                start0, CZxy, CZz, opCZ = geomCZGate(r, qubits, start0, Omega2, delay, delta)
                for idx in range(3):
                    xys[idx] += CZxy[idx]
                    zs[idx] += CZz[idx]
                op = np.dot(opCZ, op)
                #--------Ss---------
                starts, ops = [],[]
                for idx,qubit in enumerate(qubits):
                    start, C, opC = CliffordFor1(qubit, start0, idxs[6,idx])
                    xys[idx+1] += C
                    starts.append(start)
                    ops.append(opC)
                start0 = np.max(starts)
                op = np.dot(reduce(np.kron, ops), op)
        
    return start0, xys, zs, op    

#============other gates====================
def targetGateFor1(q, start, gate, alpha=None,df=None, piAmp=None, piAmpHalf=None,piAmpZ=None, cosEnv=False):
    '''basic target gates for single qubit RB process.
        return end time of the sequence, sequence, and the operator matrix of the target gate
    '''
    if alpha is None:
        alpha=q.alpha.value
    if df is None:
        df=q.piDf
    if cosEnv:
        if piAmp is None:
            piAmp=q.piAmpCosine
        if piAmpHalf is None:
            piAmpHalf=q.piAmpHalfCosine    
    else:
        if piAmp is None:
            piAmp=q.piAmp
            angle=np.pi*piAmp/q.piAmp 
        if piAmpHalf is None:
            piAmpHalf=q.piAmpHalf
            angleHalf=np.pi/2.0*piAmpHalf/q.piAmpHalf       
    if piAmpZ is None:
        piAmpZ=q.piAmpZ
        
    xy = env.NOTHING
    z = env.NOTHING
    if gate == 'reference':#sequence for gate
        op = I2
    else:
        start += q.piLen/2.0
        if cosEnv:
            if gate == 'X/2':
                xy = eh.mix(q,eh.piHalfPulseCosine(q,start, alpha=alpha, piAmp=piAmpHalf))
                op = halfX
            elif gate == 'X':
                xy = eh.mix(q,eh.piPulseCosine(q,start, alpha=alpha, piAmp=piAmp))
                op = X
            elif gate == '-X/2':
                xy = eh.mix(q,eh.piHalfPulseCosine(q,start,phase=np.pi, alpha=alpha, piAmp=piAmpHalf))
                op = mhalfX
            elif gate == '-X':
                xy = eh.mix(q,eh.piPulseCosine(q,start,phase=np.pi, alpha=alpha, piAmp=piAmp))
                op = mX
            elif gate == 'Y/2':
                xy = eh.mix(q,eh.piHalfPulseCosine(q,start,phase=np.pi/2, alpha=alpha, piAmp=piAmpHalf))
                op = halfY
            elif gate == 'Y':
                xy = eh.mix(q,eh.piPulseCosine(q,start,phase=np.pi/2, alpha=alpha, piAmp=piAmp))
                op = Y
            elif gate == '-Y/2':
                xy = eh.mix(q,eh.piHalfPulseCosine(q,start,phase=-np.pi/2, alpha=alpha, piAmp=piAmpHalf))
                op = mhalfY
            elif gate == '-Y':
                xy = eh.mix(q,eh.piPulseCosine(q,start,phase=-np.pi/2, alpha=alpha, piAmp=piAmp))
                op = mY
            elif gate == 'Z':
                z = env.rect(start-q.piLen/2.0, q.piLen, piAmpZ)
                op = Z
            else:
                raise Exception('Unknown gate operation.')
        else:
            if gate == 'X/2':
                xy = eh.mix(q,eh.piHalfPulse(q,start, alpha=alpha, angle=angleHalf))
                op = halfX
            elif gate == 'X':
                xy = eh.mix(q,eh.piPulse(q,start, alpha=alpha, angle=angle))
                op = X
            elif gate == '-X/2':
                xy = eh.mix(q,eh.piHalfPulse(q,start,phase=np.pi, alpha=alpha, angle=angleHalf))
                op = mhalfX
            elif gate == '-X':
                xy = eh.mix(q,eh.piPulse(q,start,phase=np.pi, alpha=alpha, angle=angle))
                op = mX
            elif gate == 'Y/2':
                xy = eh.mix(q,eh.piHalfPulse(q,start,phase=np.pi/2, alpha=alpha, angle=angleHalf))
                op = halfY
            elif gate == 'Y':
                xy = eh.mix(q,eh.piPulse(q,start,phase=np.pi/2, alpha=alpha, angle=angle))
                op = Y
            elif gate == '-Y/2':
                xy = eh.mix(q,eh.piHalfPulse(q,start,phase=-np.pi/2, alpha=alpha, angle=angleHalf))
                op = mhalfY
            elif gate == '-Y':
                xy = eh.mix(q,eh.piPulse(q,start,phase=-np.pi/2, alpha=alpha, angle=angle))
                op = mY
            elif gate == 'Z':
                z = env.rect(start-q.piLen/2.0, q.piLen, piAmpZ)
                op = Z
            elif gate == 'Z/2':
                z = env.rect(start-q.piLen/2.0, 10*ns, piAmpZ)
                op = halfZ
                start -= q.piLen
                start += 10*ns
                
            elif gate == '2Z/5':
                z = env.rect(start-q.piLen/2.0, 10*ns, piAmpZ)
                start += 10*ns
                start += 4*ns
                
                # z += env.rect(start-q.piLen/2.0, 10*ns, piAmpZ)
                # start += 10*ns
                # start += 4*ns
                
                # z += env.rect(start-q.piLen/2.0, 10*ns, piAmpZ)
                # start += 10*ns
                # start += 4*ns
                
                # z += env.rect(start-q.piLen/2.0, 10*ns, piAmpZ)
                # start += 10*ns
                # start += 4*ns
                
                # z += env.rect(start-q.piLen/2.0, 10*ns, piAmpZ)
                # start += 10*ns
                # start += 4*ns
                
                op = halfZ
                #op = I2
                start -= q.piLen
                
            elif gate == '2X/11':
                xy = eh.mix(q, eh.rotPulseHD(q, start-q.piLen/2.0, 0.12447, phase=0, alpha=1.471, w=10*ns))
                start += 10*ns
                
                for i in range(10):
                    xy += eh.mix(q, eh.rotPulseHD(q, start-q.piLen/2.0, 0.12447, phase=0, alpha=1.471, w=10*ns))
                    start += 10*ns
                
                # xy += eh.mix(q, eh.rotPulseHD(q, start-q.piLen/2.0, 0.12447, phase=np.pi, alpha=1.471, w=10*ns))
                # start += 10*ns
                
                op = I2
                start -= q.piLen
                
                
            elif gate == 'Z/2-Five':
                z = env.rect(start-q.piLen/2.0, 10*ns, piAmpZ)
                start += 10*ns
                start += 4*ns
                
                start += 5*ns
                xy = eh.mix(q, eh.rotPulseHD(q, start-q.piLen/2.0, 0.12447, phase=0, alpha=1.471, w=10*ns))
                start += 5*ns
                
                start += 5*ns
                xy += eh.mix(q, eh.rotPulseHD(q, start-q.piLen/2.0, 0.12447, phase=-np.pi/2, alpha=1.471, w=10*ns))
                start += 5*ns
 
                start += 5*ns
                xy += eh.mix(q, eh.rotPulseHD(q, start-q.piLen/2.0, 0.12447, phase=np.pi, alpha=1.471, w=10*ns))
                start += 5*ns
                
                start += 5*ns
                xy += eh.mix(q, eh.rotPulseHD(q, start-q.piLen/2.0, 0.12447, phase=np.pi/2, alpha=1.471, w=10*ns))
                start += 5*ns
                
                op = halfZ
                start -= q.piLen
                
            elif gate == 'Z/2-One':
                z = env.rect(start-q.piLen/2.0, 10*ns, piAmpZ)
                start += 10*ns
                start += 4*ns
                
                z += env.rect(start-q.piLen/2.0, 10*ns, 0.08298)
                start += 10*ns
                start += 4*ns
                
            elif gate == 'X/100':
                w = 2*ns
                start += w
                xy = eh.mix(q, eh.rotPulseHD(q, start-q.piLen/2.0, 0.03781, phase=0, alpha=1.432, w=w))
                start += w
                for n in range(99):
                    start += w
                    xy += eh.mix(q, eh.rotPulseHD(q, start-q.piLen/2.0, 0.03781, phase=0, alpha=1.432, w=w))
                    start += w
         
                op = X
                start -= q.piLen
                
            elif gate == 'Z/100':
                z = env.rect(start-q.piLen/2.0, 4*ns, 0.00425)
                start += 4*ns
                start += 1*ns
                for n in range(99):
                    z += env.rect(start-q.piLen/2.0, 4*ns, 0.00425)
                    start += 4*ns
                    start += 1*ns
         
                op = Z
                start -= q.piLen
          
            elif gate == 'H':
                z = env.rect(start-q.piLen/2.0, 4*ns, 0.00425)
                start += 4*ns
                start += 1*ns
                
                w = 2*ns
                start += w
                xy = eh.mix(q, eh.rotPulseHD(q, start-q.piLen/2.0, 0.03781, phase=0, alpha=1.432, w=w))
                start += w
                
                for n in range(70):
                    z += env.rect(start-q.piLen/2.0, 4*ns, 0.00425)
                    start += 4*ns
                    start += 1*ns
 
                    start += w
                    xy += eh.mix(q, eh.rotPulseHD(q, start-q.piLen/2.0, 0.03781, phase=0, alpha=1.432, w=w))
                    start += w
                op = XZ
                start -= q.piLen           
                
            
            else:
                raise Exception('Unknown gate operation.')
        start += q.piLen/2.0
    return start, xy, z, op

    
def tunnelingv1(qubits, data):# Chao Song 2015-02-08
    prob = []
    for q in qubits:
        Is = np.asarray(data[0][0][0])
        Qs = np.asarray(data[0][0][1])    
        sigs = Is + 1j*Qs
        center_0 = q['center|0>'][0] + 1j*q['center|0>'][1]
        center_1 = q['center|1>'][0] + 1j*q['center|1>'][1]
        center = (center_0+center_1)/2.0
        theta = np.angle(center_0-center_1)
        sigs = (sigs-center)*np.exp(-1j*theta)
        total = len(Is)
        # tunnel = 0
        # for sig in sigs:
            # if np.real(sig)<0:
                # tunnel += 1
        tunnel = np.sum(np.real(sigs)<q['criticalI'])
        prob.append(float(tunnel)/float(total))        
    return prob

def tunneling_multilevel(qubit, data):
    q = qubit
    Is = np.asarray(data[0][0][0])
    Qs = np.asarray(data[0][0][1])    
    sigs = Is + 1j*Qs
    center_0 = q['center|0>'][0] + 1j*q['center|0>'][1]
    center_1 = q['center|1>'][0] + 1j*q['center|1>'][1]
    center_2 = q['center|2>'][0] + 1j*q['center|2>'][1]
    distance_0 = np.abs(sigs - center_0)
    distance_1 = np.abs(sigs - center_1)
    distance_2 = np.abs(sigs - center_2)
    total = len(Is)
    tunnel_0 = 0
    tunnel_1 = 0
    tunnel_2 = 0
    
    for idx in np.arange(len(sigs)):
        if distance_0[idx]<distance_1[idx] and distance_0[idx]<distance_2[idx]: tunnel_0 += 1
        if distance_1[idx]<distance_0[idx] and distance_1[idx]<distance_2[idx]: tunnel_1 += 1
        if distance_2[idx]<distance_1[idx] and distance_2[idx]<distance_0[idx]: tunnel_2 += 1
    prob0 = float(tunnel_0)/float(total)
    prob1 = float(tunnel_1)/float(total)
    prob2 = float(tunnel_2)/float(total)
    return prob0, prob1, prob2

def stateFidelity(q, result, rho_ideal, doPlot=False):
    Qk = np.reshape(result[1:],(3,2))
    fidC = np.matrix([[q.measureF0, 1-q.measureF1],[1-q.measureF0,q.measureF1]])
    fidCinv = fidC.I
    dataC = Qk*0
    for i in range(len(Qk[:,0])):
        dataC[i,:] = np.dot(fidCinv,Qk[i,:])
    rho_cal = tomo.qst(dataC,'tomo')
    if doPlot:
        plotRho(rho_cal)
    print np.round(rho_cal,3)
    return np.trace(np.dot(np.abs(rho_cal),np.abs(rho_ideal))),rho_cal

def runSingleQubitTomo(s, measure=1): 
    for angle in np.linspace(0,np.pi,21):
        singleQubitTomo(s, 10, angle, phase=0.0, measure=measure, stats=1200, des=' angle=%.2f'%angle)
    # for angle in np.linspace(0,np.pi,21):
        # singleQubitTomo(s, 10, angle, phase=0.0, measure=measure, stats=1200, df=0*MHz, des=' angle=%.2f'%angle)
        
    # for angle in np.linspace(0,np.pi,21):
        # singleQubitTomo(s, 10, angle, phase=0.0, measure=measure, stats=1200, dfTomo=0*MHz, des=' angle=%.2f'%angle)
    # for angle in np.linspace(0,np.pi,21):
        # singleQubitTomo(s, 10, angle, phase=0.0, measure=measure, stats=1200, df=0*MHz, dfTomo=0*MHz, des=' angle=%.2f'%angle)

def singleQubitTomo(sample, repetition=10, angle=np.pi/2, phase=0.0,measure=0, stats=1500,delay=0.0*ns,df=None,dfTomo=None,
               save=True, name='single qubit Tomo', collect=False, noisy=True, des='',rho_ideal=None): # by Chao Song 2015-03-20
    '''single qubit tomography
       
        ''' 
    sample, qubits = util.loadQubits(sample)
    paraAmp = qubits[0]
    q = qubits[measure]
    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    if df is None:
        df = q.piDf
    if dfTomo is None:
        dfTomo = q.piDf
    deps = []
    ops = ['I','X','Y']
    labels = ['0','1']
    for op1 in ops:
        for l1 in labels: 
            deps.append((op1,'P|'+l1+'>',''))
    kw = {'stats': stats,
          'angle': angle,
          'phase': phase,
          'df':df,
          'dfTomo':dfTomo,
          }
    dataset = sweeps.prepDataset(sample, name+des, axes, deps, measure=measure, kw=kw)
    fidCinv = np.matrix([[q.measureF0, 1-q.measureF1],[1-q.measureF0,q.measureF1]]).I
    def func(server,curr):
        reqs = []
        for qubit in qubits:
            qubit['readout fc'] = q['readout fc']
            qubit['fc'] =q['fc']
        q['adc demod frequency'] = q['qnd_readout frequency'] - q['readout fc']
        
        for op1 in ops:        
            start = 0*ns
            q.xy = env.NOTHING
            q.z = env.NOTHING
            start += q.piLen/2.0
            q.xy += eh.mix(q,eh.rotPulseHD(q,start,angle=angle,phase=phase), piDf=df)
            start += q.piLen/2.0+delay
            start += q.piLen/2.0
            if op1 == 'I':
                top1 = q.piLen
            elif op1 == 'X':
                top1 = q.piLen
                q.xy += eh.mix(q, eh.rotPulseHD(q, start, angle=np.pi/2, phase=0.0), piDf=dfTomo)
            elif op1 == 'Y':
                top1 = q.piLen
                q.xy += eh.mix(q, eh.rotPulseHD(q, start, angle=np.pi/2, phase=np.pi/2), piDf=dfTomo)     
            start += top1
            q.rr = eh.QND_readoutPulse_ring(q,start)
            for qubit in qubits:
                qubit['adc_start_delay'] = q['adc_start_delay']
                qubit['adc filterStart'] = start + q['qnd_ringLen']+q['filterStartBase']
                qubit['adc filterStop'] = qubit['adc filterStart'] + q['qnd_readoutLen']
            q['readout'] = True
            
            data = yield FutureList([runQ(server, qubits, stats, raw=True)])
            
            prob = tunnelingv1([q], data)[0]
            if noisy: 
                print curr, op1,1-prob,prob
                print 'corrected prob:',np.dot(fidCinv,np.array([1-prob,prob]).T)
            reqs.append([1-prob, prob])
            
        returnValue(np.hstack(reqs))
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=False)
    result = np.mean(result,axis=0)
        
    sigmaX = np.array([[0, 1], [1, 0]], dtype=complex)
    sigmaY = np.array([[0, -1j], [1j, 0]], dtype=complex)
    axis0 = sigmaX*np.cos(phase)+sigmaY*np.sin(phase)
    rotationOpt = scipy.linalg.expm(-1j*angle/2.0*axis0)
    rho_g = np.array([[1,0],[0,0]])
    rho_ideal = np.dot(np.dot(rotationOpt,rho_g),rotationOpt.conjugate().T)
    print 'rho_ideal:\n',rho_ideal
    if rho_ideal is not None: 
        fidelity,rho_cal = stateFidelity(q, result, rho_ideal)
        print 'Fidelity: %g'%fidelity
    if collect:
        return rho_cal

def targetGateForN(qubits, start, gates):
    '''basic target gates for N qubit RB process.
        return end time of the sequence, sequence, and the operator matrix of the target gate
    '''
    qubitNum = len(qubits)
    xys = [env.NOTHING]*qubitNum
    zs = [env.NOTHING]*qubitNum
    start0 = start
    starts, ops = [],[]
    for idx,qubit in enumerate(qubits):
        start, xy, z, op = targetGateFor1(qubit, start0, gates[idx])
        xys[idx] += xy
        zs[idx] += z
        starts.append(start)
        ops.append(op)
    start = np.max(starts)
    op = reduce(np.kron, ops)
    return start, xys, zs, op
        
def geomCZGate(r, qubits, start, Omega2, delay, delta):
    '''
        geometric CZ phase gate.
        apply drive to resonator while biasing qubits to prefered frequency.
        return sequences of r and qubits.
    '''
    qubitNum = len(qubits)
    
    op = np.diag([1.0]*2**qubitNum)
    op[0,0] = -1.0
    
    xys = [env.NOTHING]*(qubitNum+1)
    zs = [env.NOTHING]*(qubitNum+1)
    sb_freq = r.fr+delta-r.fc
    r.spectroscopyLen = delay
    r.spectroscopyAmp = 0.2*np.pi*Omega2**0.5/r.scaleFactor.value 
    zpa_seps = [qubit.zpa_sep for qubit in qubits]
    #-----geometric phase--------------
    zpa_sep = np.max(zpa_seps)
    zLen = 2*r.driveSlopeLen + delay + 2*zpa_sep
    for idx, qubit in enumerate(qubits):
        zs[idx+1] += env.rect(start+qubit.zpa_sep, zLen-2*qubit.zpa_sep, qubit.zpa_exp)
    start += zpa_sep
    start += r.driveSlopeLen
    xys[0] += eh.spectroscopyPulse(r, start, sb_freq)
    start += r.driveSlopeLen
    start += delay
    start += zpa_sep
    #---------compensation pulse----------------------------------
    for idx, qubit in enumerate(qubits):
        zs[idx+1] += env.rect(start, 10*ns, qubit.zpa_comp)
    start += 10*ns
    
    return start, xys, zs, op


    
def recoveryGateFor1(q, start, op, cosEnv=False):
    start += q.piLen/2.0
    xy = env.NOTHING
    if np.abs(op[0,1])<0.01:
        pass
    elif np.abs(op[0,1])>0.99:
        if np.abs(np.dot(X,op)[0,1])<0.01:
            correctPhase = 0
        elif np.abs(np.dot(mX,op)[0,1])<0.01:
            correctPhase = np.pi
        elif np.abs(np.dot(Y,op)[0,1])<0.01:
            correctPhase = np.pi/2.0
        elif np.abs(np.dot(mY,op)[0,1])<0.01:
            correctPhase = -np.pi/2.0
        else:
            raise Exception('cannot find the final correct rotation.')
        if cosEnv:  
            xy = eh.mix(q,eh.piPulseCosine(q,start,phase=correctPhase))
        else:
            xy = eh.mix(q,eh.piPulse(q,start,phase=correctPhase))
    else:
        if np.abs(np.dot(halfX,op)[0,1])<0.01:
            correctPhase = 0
        elif np.abs(np.dot(mhalfX,op)[0,1])<0.01:
            correctPhase = np.pi
        elif np.abs(np.dot(halfY,op)[0,1])<0.01:
            correctPhase = np.pi/2.0
        elif np.abs(np.dot(mhalfY,op)[0,1])<0.01:
            correctPhase = -np.pi/2.0
        else:
            raise Exception('cannot find the final correct rotation.')
        if cosEnv:
            xy = eh.mix(q,eh.piHalfPulseCosine(q,start,phase=correctPhase))
        else:
            xy = eh.mix(q,eh.piHalfPulse(q,start,phase=correctPhase))
    start += q.piLen/2.0
    return start, xy    
    
def recoveryGateFor2(r, qubits, start, data, Omega2, delay, delta, op, check=False, tbuffer=0*ns):
    #extract infomation from data
    c1s = data[0,:]
    czs = data[1,:]
    c2s = data[2,:]
    #-------------------------------
    qubitNum = len(qubits)+1
    xys = [env.NOTHING]*qubitNum
    zs = [env.NOTHING]*qubitNum
    start0 = start
    
    start, xy1, op1 = CliffordForN(qubits, start, c1s)
    start += tbuffer
    if czs[1]:
        print '\n recover: add CZ'
        
        start, xy2, z2, op2 = geomCZGate(r, qubits, start, Omega2, delay, delta)
        start += tbuffer
    else:
        op2 = np.diag([1.0]*4)
        xy2, z2 = [env.NOTHING]*3, [env.NOTHING]*3
    start, xy3, op3 = CliffordForN(qubits, start, c2s)
    op_final = reduce(np.dot, [op3,op2,op1,op])
    state_initial = [1.0, 0, 0, 0]
    state_final = np.dot(op_final, state_initial)
    print 'check recover: ',np.round(np.abs(state_final),1)
    if check:
        print '####################'
        print np.round(op, 1)
        print np.round(op1, 1)
        print np.round(op2, 1)
        print np.round(op3, 1)
        print np.round(op_final, 1)
    
    for idx in range(qubitNum-1):
        xys[idx+1] += xy1[idx]
        xys[idx+1] += xy3[idx]
    for idx in range(qubitNum):
        xys[idx] += xy2[idx]
        zs[idx] += z2[idx]
    return start, xys, zs

def run_rb1(s, measure=0,stats=1200):
    rep = 50
    gates = ['X', 'X/2', 'Y', 'Y/2', '-X', '-X/2', '-Y', '-Y/2']
    #randomizedBenchmarking_1q(s, range(0,120,3), rep,stats=stats, gate='reference', measure=measure)
    randomizedBenchmarking_1q(s, None, rep,stats=stats, gate='reference', measure=measure)
    for gate in gates:
        #randomizedBenchmarking_1q(s, range(0,90,3), rep,stats=stats, gate=gate, measure=measure)
        randomizedBenchmarking_1q(s, None, rep,stats=stats, gate=gate, measure=measure)
def overnight20161006(s):
    # run_rb1(s, measure=0)
    run_rb1(s, measure=0)
    
def run_rb2(s):
    idx = 0
    while idx < 10:
        idx += 1
        try:
            randomizedBenchmarking_2q_full(s, range(9), 20, gate='reference', measure=[2,0,3], save=True, Omega2=7.2)
            randomizedBenchmarking_2q_full(s, range(6), 20, gate='CZ', measure=[2,0,3], save=True, Omega2=7.2)
        except:
            idx -= 1
#==================main=================
def randomizedBenchmarking_1q(sample, m=None, k = 40, gate='X/2', tbuffer=0*ns, measure=0, measureLevel=1, stats=600, name='Randomized Benchmarking 1q ', save=True, noisy=True, collect=False):
    '''
    Single Qubit Randomized Benchmarking.
    Parameters:
    m(list):the maximum number of the Cliffords;
    k(integer):the nummber of different random sequence;
    gate(string):the gate to marking;
    ''' 
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    if m is None:
        if gate == 'reference':
            m = np.hstack((np.arange(1, 11) , np.array([10+np.floor(10**i) for i in np.linspace(0, 2.5, 30)])))# max=110
            
        else:
            #m = np.hstack((np.arange(1, 11) , np.array([10+np.floor(10**i) for i in np.linspace(0, 1.8, 30)])))# max=110
       # m = np.hstack((np.arange(1, 11) , np.array([10+np.floor(10**i) for i in np.linspace(0, 2.3, 30)])))
        #m = np.hstack((np.arange(1, 11) , np.array([10+np.floor(10**i) for i in np.linspace(0, 2.46, 30)])))# max=298
       # m = np.hstack((np.arange(1, 11) , np.array([10+np.floor(10**i) for i in np.linspace(0, 2.14, 30)])))# max=150
       # m = np.hstack((np.arange(1, 11) , np.array([10+np.floor(10**i) for i in np.linspace(0, 2.00, 30)])))# max=110
        
             m = np.hstack((np.arange(1, 11) , np.array([10+np.floor(10**i) for i in np.linspace(0, 2.3, 30)])))# max=90
        
        # m = np.hstack((np.arange(1, 11) ,np.arange(11, 80, 5), np.arange(80, 200, 15)))
        m = np.unique(m)
        m=np.asarray(map(int,m))
    axes = [(np.arange(k), 'k'), (m, 'm')]
    
    kw = {'stats':stats,
          'tbuffer':tbuffer,
          'gate':gate,
          'measure':measure,
         }
    
    if np.iterable(m):
        M = int(np.max(m))
    else:
        M = m
    if measureLevel == 1:
        deps = [('Prob', '|0>', ''), ('Prob', '|1>', '')]   #operations for detection
    elif measureLevel == 2:
        deps = [('Prob', '|0>', ''), ('Prob', '|1>', ''), ('Prob', '|2>', '')] 
    else:
        raise Exception('How many levels do you want to measure??')
    randomNumbers=numpy.random.randint(0,24,(k,M))
    dataset = sweeps.prepDataset(sample, name+gate, axes, deps, measure=measure, kw=kw) 
    
    def func(server, k, m):    #define the sequence for randomized benchmarking
        for qubit in qubits:
            qubit['adc_start_delay'] = q['adc_start_delay']
            qubit['readout DAC start delay'] = 0*ns 
            qubit['readout fc'] = q['readout fc']
            qubit['fc'] = q['fc']
        q['adc demod frequency'] = q['qnd_readout frequency']- q['readout fc']
        start = 0
        q.xy = env.NOTHING
        q.z = env.NOTHING
        op = np.diag([1.0,1.0])
        for idx in range(m):
            randomNumber = randomNumbers[k,idx]
            #---------clifford gates------
            start, xy, operator = CliffordFor1(q, start, randomNumber)
            q.xy += xy
            op = np.dot(operator,op)
            start += tbuffer
            #---------target gate------
            start, xy, z, operator = targetGateFor1(q, start, gate)
            q.xy += xy
            q.z += z
            op = np.dot(operator,op)
            start += tbuffer
        #---------recovery gates------
        start, xy = recoveryGateFor1(q, start, op)
        q.xy += xy
        #----------readout------------
        q.rr = eh.QND_readoutPulse_ring(q,start)#readout
        for qubit in qubits:
            qubit['adc filterStart'] = start + q['qnd_ringLen']+q['filterStartBase']
            qubit['adc filterStop'] = qubit['adc filterStart'] + q['qnd_readoutLen']
        q['readout'] = True
        data = yield FutureList([runQ(server, qubits, stats, raw=True)])
        if measureLevel == 1:
            prob = mp.tunnelingv1([q], data)[0]
            if noisy:
                print np.round([k,m,prob],3)
            probs = [1.0-prob,prob]
        elif measureLevel == 2:
            probs = mp.tunneling_multilevel(q, data)
            if noisy:
                print k,m
                print np.round(probs,3)
        else:
            raise Exception('How many levels do you want to measure??')
        returnValue(probs)

    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if collect:
        return data        

def measure_error_vs_alpha(s, measure):
    m = range(1,351,11)
    alphas = np.arange(0,1.51,0.1)
    k = 30
    dfs = np.array([1.47, 0.92, 1.061, 0.510, -0.052, -0.656, -1.229, -1.877, -2.453, -3.026, -3.638, -4.329, -5.001, -5.683, -6.334, -6.963])*MHz
    for idx, alpha in enumerate(alphas):
        orbit_1q(s, m, k=k, alpha=alpha, measure=measure, gate='reference', measureLevel=2)
    # for idx, alpha in enumerate(alphas[::2]):
        # print alpha, dfs[::2][idx]
        # orbit_1q(s, m, k=k, alpha=alpha, df=dfs[::2][idx], measure=measure, gate='reference', measureLevel=2)

def overnoon20170112(s):
    gates=['reference','X','Y','-X','-Y','X/2','-X/2','-Y/2','Y/2']
    # gates=['reference','X/2','Y','-X','-Y','-X/2','-Y/2','Y/2']
    k = 20
    for idx in range(5):
        for gate in gates:
            orbit_1q(s, k=k, gate=gate)
        
def overnight20170112(s,measure=0):
    
    gates=['reference','X','Y','-X','-Y','X/2','-X/2','Y/2','-Y/2']
    # gates=['reference','X/2','Y','-X','-Y','-X/2','-Y/2','Y/2']
    k = 20
    for idx in range(5):
        for gate in gates:
            orbit_1q(s, k=k, gate=gate,measure=measure)
    m = np.hstack((np.arange(1, 11) , np.array([10+np.floor(10**i) for i in np.linspace(0, 2.00, 30)])))# max=110
    orbit_1q(s, k=30, m=m, gate='reference',measure=measure)
   
def overnight20170113(s,measure=0):
    
    gates=['reference','X','Y','-X','-Y','X/2','-X/2','Y/2','-Y/2']
    # gates=['reference','X/2','Y','-X','-Y','-X/2','-Y/2','Y/2']
    k = 20
    for idx in range(5):
        for gate in gates:
            orbit_1q(s, k=k, gate=gate,measure=measure)
    m = np.hstack((np.arange(1, 11) , np.array([10+np.floor(10**i) for i in np.linspace(0, 2.00, 30)])))# max=110
    orbit_1q(s, k=30, m=m, gate='reference',measure=measure)   
    
    s.q4.piFWHM=8*ns
    s.q4.piFWHMHalf=8*ns
    s.q4.piAmp=0.868
    s.q4.piAmpHalf=0.4306
    s.q4.alpha=1.369
    s.q4.alphaHalf=1.373
    
    for idx in range(5):
        for gate in gates:
            orbit_1q(s, k=k, gate=gate,measure=measure)
    m = np.hstack((np.arange(1, 11) , np.array([10+np.floor(10**i) for i in np.linspace(0, 2.00, 30)])))# max=110
    orbit_1q(s, k=30, m=m, gate='reference',measure=measure) 

def overnoon20170113(s,measure=0):
    
    # gates=['reference','X','Y','-X','-Y','X/2','-X/2','Y/2','-Y/2']
    gates=['reference','X','Y','-X','-Y','X/2','-X/2','Y/2','-Y/2']
    k = 20
    for idx in range(5):
        for gate in gates:
            orbit_1q(s, k=k, gate=gate,measure=measure)
    m = np.hstack((np.arange(1, 11) , np.array([10+np.floor(10**i) for i in np.linspace(0, 2.00, 30)])))# max=110
    # orbit_1q(s, k=30, m=m, gate='reference',measure=measure)
 
def overnight20161011(cxn, switchSession):
    m = range(1,351,11)
    alphas = np.arange(0,1.51,0.1)
    k = 30
    for idx, alpha in enumerate(alphas):
        cxn.registry.cd(['','Chao'])
        cxn.registry.set('sample',['20161011_optimize_pulses'])
        s = switchSession(user='Chao')
        time.sleep(1)
        orbit_1q(s, m, k=k, alpha=alpha, df=0*MHz, measure=0, gate='reference', measureLevel=2)
        
        cxn.registry.cd(['','Chao'])
        cxn.registry.set('sample',['20161011_controlGeoPhase_3q_e_old'])
        s = switchSession(user='Chao')
        time.sleep(1)
        mp.T1_f10_2d(s, delay=st.r[0:2:0.05,us], f10s=st.r[5.8:5.85:0.005,GHz], measure=1, stats=900, des=' 2D')
    
    cxn.registry.cd(['','Chao'])
    cxn.registry.set('sample',['20161011_optimize_pulses'])
    s = switchSession(user='Chao')  
    alphas = [0, 0.5, 1.0, 1.5]
    dfs = [5.821, 2.929, 0.043, -2.713 ]
    for idx, alpha in enumerate(alphas):
        orbit_1q(s, m, k=k, alpha=alpha, df=dfs[idx]*MHz, measure=0, gate='reference', measureLevel=2)
 
def overnight20161013(cxn, switchSession):
    cxn.registry.cd(['','Chao'])
    cxn.registry.set('sample',['20161011_optimize_pulses'])
    s = switchSession(user='Chao')
    gates=['reference','X','Y','-X','-Y','X/2','-X/2','-Y/2','Y/2']
    
    k = 40
    for gate in gates:
        if gate == 'reference':
            m = range(1,300,6)
        else:
            m = range(1,180,6)
        orbit_1q(s,m=m,k=k,alpha=0,df=0.642*MHz,gate=gate)
    
    cxn.registry.cd(['','Chao'])
    cxn.registry.set('sample',['20161011_controlGeoPhase_3q_e_old'])
    s = switchSession(user='Chao')
    for i in range(10):
        mp.T1_f10_2d(s, delay=st.r[0:2:0.05,us], f10s=st.r[5.8:5.85:0.005,GHz], measure=1, stats=900, des=' 2D')
 
def overnight20161015(s):
    gates=['reference','X','Y','-X','-Y','X/2','-X/2','-Y/2','Y/2']
    k = 20
    for idx in range(5):
        for gate in gates:
            orbit_1q(s, k=k, gate=gate)
    # mp.T1(s, delay=None,zpa=st.r[-0.05:0.05:0.001], measure=0, stats=900, des=' 2D')
    
    
def overnight20161016(cxn, switchSession):
    cxn.registry.cd(['','Chao'])
    cxn.registry.set('sample',['20161011_optimize_pulses'])
    s = switchSession(user='Chao')
    run_orbit_1q(s,k=50, piAmp=None, alpha=None,piAmpHalf=None, df=st.r[-1:1:0.05,MHz],gate='reference', ms=[1,50,100], des=' df')
    run_orbit_1q(s,k=50, piAmp=None, alpha=None,piAmpHalf=st.r[-0.05:0.05:0.002], df=None,gate='reference', ms=[1,50,100], des=' piAmpHalf')
    run_orbit_1q(s,k=50, piAmp=st.r[-0.05:0.05:0.002], alpha=None,piAmpHalf=None, df=None,gate='reference', ms=[1,50,100], des=' piAmp')
    
    cxn.registry.cd(['','Chao'])
    cxn.registry.set('sample',['20161011_controlGeoPhase_3q_e_old'])
    s = switchSession(user='Chao')
    for i in range(10):
        mp.T1_f10_2d(s, delay=st.r[0:2:0.01,us], f10s=st.r[5.8:5.85:0.005,GHz], measure=1, stats=900, des=' 2D')
def run_orbit_1q(s, k=20, alpha=None, piAmp=None, piAmpHalf=None, piAmpZ=None, df=None, measure=0, ms=[1,50,100],gate='X', des='', cosEnv=False):
    for m in ms:
        orbit_1q(s, m, k=k, alpha=alpha, piAmp=piAmp, piAmpHalf=piAmpHalf, piAmpZ=piAmpZ, df=df, measure=measure, gate=gate, des=des, cosEnv=cosEnv)

def run_orbit_1q_v2(s, k= 20, alpha=None, piAmp=None, piAmpHalf=None, df=None, measure=0, ms=[1,50,100],gate='X'):
    for m in ms:
        orbit_1q_v2(s, m, k=k, alpha=alpha, piAmp=piAmp, piAmpHalf=piAmpHalf, df=df, measure=measure, gate=gate)

    
    
def orbit_1q(sample, m=None, k = 40, alpha=None, piAmp=None, piAmpHalf=None, piAmpZ=None, df=None, gate='X', tbuffer=0*ns, measure=0, 
             measureLevel=1, stats=600, name='orbit 1q ', save=True, noisy=True, collect=False, cosEnv=False, des=''):
    '''
    alpha/piAmp/piAmpHalf/piAmpZ/df should be the difference from their key value
    optimize pi pulse
    Single Qubit Randomized Benchmarking.
    Parameters:
    m(list):the maximum number of the Cliffords;
    k(integer):the nummber of different random sequence;
    gate(string):the gate to marking;
    ''' 
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    if alpha is not None:
        origiAlpha = q.alpha
    if cosEnv:
        if piAmp is not None:
            origiPiAmp = q.piAmpCosine
        if piAmpHalf is not None:
            origiPiAmpHalf = q.piAmpHalfCosine
    else:
        if piAmp is not None:
            origiPiAmp = q.piAmp
        if piAmpHalf is not None:
            origiPiAmpHalf = q.piAmpHalf
    
    if df is not None:
        origiDf = q.piDf
    if piAmpZ is not None:
        origiPiAmpZ = q.piAmpZ
    if m is None:
        if gate == 'reference':
            m = np.hstack((np.arange(1, 11) , np.array([10+np.floor(10**i) for i in np.linspace(0, 2.5, 30)])))# max=110
            
        else:
            #m = np.hstack((np.arange(1, 11) , np.array([10+np.floor(10**i) for i in np.linspace(0, 1.8, 30)])))# max=110
       # m = np.hstack((np.arange(1, 11) , np.array([10+np.floor(10**i) for i in np.linspace(0, 2.3, 30)])))
        #m = np.hstack((np.arange(1, 11) , np.array([10+np.floor(10**i) for i in np.linspace(0, 2.46, 30)])))# max=298
       # m = np.hstack((np.arange(1, 11) , np.array([10+np.floor(10**i) for i in np.linspace(0, 2.14, 30)])))# max=150
       # m = np.hstack((np.arange(1, 11) , np.array([10+np.floor(10**i) for i in np.linspace(0, 2.00, 30)])))# max=110
        
             m = np.hstack((np.arange(1, 11) , np.array([10+np.floor(10**i) for i in np.linspace(0, 2.3, 30)])))# max=90
        
        # m = np.hstack((np.arange(1, 11) ,np.arange(11, 80, 5), np.arange(80, 200, 15)))
        m = np.unique(m)
    axes = [(np.arange(k), 'k'), (m, 'm'), (alpha, 'alpha'), (piAmp, 'piAmp'), (piAmpHalf, 'piAmpHalf'), (piAmpZ, 'piAmpZ'), (df, 'df')]
    
    kw = {'stats':stats,
          'tbuffer':tbuffer,
          'gate':gate,
          'measure':measure,
          
         }
    
    if np.iterable(m):
        M = int(np.max(m))
    else:
        M = int(m)
    if measureLevel == 1:
        deps = [('Prob', '|0>', ''), ('Prob', '|1>', '')]   #operations for detection
    elif measureLevel == 2:
        deps = [('Prob', '|0>', ''), ('Prob', '|1>', ''), ('Prob', '|2>', '')] 
    else:
        raise Exception('How many levels do you want to measure??')
    # randomNumbers=numpy.random.randint(0,24,(k,M))
    # print randomNumbers
    # kw['randomNumbers'] = randomNumbers
    windowsBackup = q['adcReadoutWindows'].copy()
    powerBack = q['qnd_readout power']
    dataset = sweeps.prepDataset(sample, name+gate+des, axes, deps, measure=measure, kw=kw) 
    def func(server, k, m, alpha, piAmp, piAmpHalf, piAmpZ, df):    #define the sequence for randomized benchmarking
        randomNumbers=numpy.random.randint(0,24,M)
        for qubit in qubits:
            qubit['adc_start_delay'] = q['adc_start_delay']
            qubit['readout DAC start delay'] = 0*ns 
            qubit['readout fc'] = q['readout fc']
            qubit['fc'] = q['fc']
        q['adc demod frequency'] = q['qnd_readout frequency']- q['readout fc']
        
        start = 0
        q.xy = env.NOTHING
        q.z = env.NOTHING
        q.rr = env.NOTHING
        if alpha is not None:
            q.alpha = origiAlpha + alpha
        if cosEnv:
            if piAmp is not None:
                q.piAmpCosine = origiPiAmp + piAmp
            if piAmpHalf is not None:
                q.piAmpHalfCosine = origiPiAmpHalf + piAmpHalf
        else:
            if piAmp is not None:
                q.piAmp = origiPiAmp + piAmp
            if piAmpHalf is not None:
                q.piAmpHalf = origiPiAmpHalf + piAmpHalf
        if df is not None:
            q.piDf = origiDf + df
        if piAmpZ is not None:
            q.piAmpZ = origiPiAmpZ + piAmpZ
        op = np.diag([1.0,1.0])
        for idx in range(int(m)):
            randomNumber = randomNumbers[idx]
            # --------clifford gates------
            start, xy, operator = CliffordFor1(q, start, randomNumber, cosEnv=cosEnv)
            q.xy += xy
            op = np.dot(operator,op)
            start += tbuffer
            # --------target gate------
            start, xy, z, operator = targetGateFor1(q, start, gate, cosEnv=cosEnv)
            q.xy += xy
            q.z += z
            op = np.dot(operator,op)
            start += tbuffer
        # --------recovery gates------
        start, xy = recoveryGateFor1(q, start, op, cosEnv=cosEnv)
        q.xy += xy
        # ---------readout------------
        tShift = (np.floor(start['ns']//4)+1)*4
        q.rr += env.shift(eh.QND_readoutPulse_ring(q, 0), dt=(tShift))
        q['readout'] = True
        
        for qubit in qubits:
            for key in qubit['adcReadoutWindows']:
                window = windowsBackup[key]
                shiftedWindow = [window[0]+tShift, window[1]+tShift]
                qubit['adcReadoutWindows'][key] = shiftedWindow
        
        data = yield FutureList([runQ(server, qubits, stats, raw=True)])
        data = np.asarray(data)
        print np.shape(data)
        probs = mp.tunnelingv1([q],data)
        #probs = mp.tunnelingQubits([q],data)
        print np.shape(probs)
        if noisy:
            print np.round([k,m,probs[0]],3)
        returnValue(probs)

    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if collect:
        return data
        

def orbitXYNM(sample, measure, paramNames = ['alpha','piAmp','piAmpHalf','piDf'],
               xtol = 0.0001, ftol = 0.001,
               nonzdelt = 0.02, zdelt = 0.02,
               name = 'RB Nelder Mead XY',
               m = 50, k = 20, stats=900,
               maxfun=None,
               interleaved=None,
               collect=False,
               save = True, noisy = True):
 
    
    
    sample, qubits = util.loadQubits(sample)
    sample, devs = util.loadDevices(sample)
    q = qubits[measure]
    qubitAndParamNames = popt.addQubitNamesToParamNames(paramNames, q)
    
    origiPara = {}
    for para in paramNames:
        origiPara[para] = q[para]
    
    param = popt.Parameters(qubitAndParamNames, devs, sample.config[measure])
    
    axes, deps, inputs = param.makeInputsAxesDeps()
    kw = {'stats': stats, 'qubitAndParamNames':qubitAndParamNames}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw) 
    
    
    def func(server, args):#return only p0
        for qubit in qubits:
            qubit['adc_start_delay'] = q['adc_start_delay']
            qubit['readout DAC start delay'] = 0*ns 
            qubit['readout fc'] = q['readout fc']
            qubit['fc'] = q['fc']
        q['adc demod frequency'] = q['qnd_readout frequency']- q['readout fc']
        randomNumbers=numpy.random.randint(0,24,m)
        for idx, para in enumerate(paramNames):
            q[para] = origiPara[para]*(1+args[idx])
        print [q[para].value for para in paramNames]
        start = 0
        q.xy = env.NOTHING
        q.z = env.NOTHING
        op = np.diag([1.0,1.0])
        for idx in range(m):
            randomNumber = randomNumbers[idx]
            # --------clifford gates------
            start, xy, operator = CliffordFor1(q, start, randomNumber)
            q.xy += xy
            op = np.dot(operator,op)
            # --------target gate------
            start, xy, z, operator = targetGateFor1(q, start, 'reference')
            q.xy += xy
            q.z += z
            op = np.dot(operator,op)
        # --------recovery gates------
        start, xy = recoveryGateFor1(q, start, op)
        q.xy += xy
        # ---------readout------------
        q.rr = eh.QND_readoutPulse_ring(q,start)#readout
        for qubit in qubits:
            qubit['adc filterStart'] = start + q['qnd_ringLen']
            qubit['adc filterStop'] = qubit['adc filterStart'] + q['qnd_readoutLen']
        q['readout'] = True
        data = yield FutureList([runQ(server, qubits, stats, raw=True)])
        prob = mp.tunnelingv1([q], data)[0]
        if noisy:
            print np.round([k,m,prob],3)
            
        returnValue([prob])
    
    funcWrapper = popt.makeFuncWrapper(k, func, axes, deps, measure, sample)
    output = sweeps.fmin(funcWrapper, inputs, 
                         dataset, xtol = xtol, ftol = ftol, nonzdelt = nonzdelt, 
                         zdelt = zdelt, maxfun=maxfun)
    
    return output

def orbitZNM(sample, measure, paramNames = ['piAmpZ', 'settlingAmplitudes', 'settlingRates'],
               xtol = 0.0001, ftol = 0.001,
               nonzdelt = 0.02, zdelt = 0.02,
               name = 'RB Nelder Mead Z',
               m = 50, k = 20, stats=900,
               maxfun=None,
               interleaved=None,
               collect=False,
               save = True, noisy = True):
 
    
    
    sample, qubits = util.loadQubits(sample)
    sample, devs = util.loadDevices(sample)
    q = qubits[measure]
    qubitAndParamNames = popt.addQubitNamesToParamNames(paramNames, q)
    
    origiPara = {}
    for para in paramNames:
        origiPara[para] = q[para]
    
    param = popt.Parameters(qubitAndParamNames, devs, sample.config[measure])
    
    axes, deps, inputs = param.makeInputsAxesDeps()
    kw = {'stats': stats, 'qubitAndParamNames':qubitAndParamNames}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw) 
    
    
    def func(server, args):#return only p0
        for qubit in qubits:
            qubit['adc_start_delay'] = q['adc_start_delay']
            qubit['readout DAC start delay'] = 0*ns 
            qubit['readout fc'] = q['readout fc']
            qubit['fc'] = q['fc']
        q['adc demod frequency'] = q['qnd_readout frequency']- q['readout fc']
        randomNumbers=numpy.random.randint(0,24,m)
        for idx, para in enumerate(paramNames):
            # print(type(para),type(idx))
            q[para] = origiPara[para]*(1+args[idx])
        print [q[para].value for para in paramNames]
        start = 0
        q.xy = env.NOTHING
        q.z = env.NOTHING
        op = np.diag([1.0,1.0])
        for idx in range(m):
            randomNumber = randomNumbers[idx]
            # --------clifford gates------
            start, xy, operator = CliffordFor1(q, start, randomNumber)
            q.xy += xy
            op = np.dot(operator,op)
            # --------target gate------
            start, xy, z, operator = targetGateFor1(q, start, 'Z')
            q.xy += xy
            q.z += z
            op = np.dot(operator,op)
        # --------recovery gates------
        start, xy = recoveryGateFor1(q, start, op)
        q.xy += xy
        # ---------readout------------
        q.rr = eh.QND_readoutPulse_ring(q,start)#readout
        for qubit in qubits:
            qubit['adc filterStart'] = start + q['qnd_ringLen']
            qubit['adc filterStop'] = qubit['adc filterStart'] + q['qnd_readoutLen']
        q['readout'] = True
        data = yield FutureList([runQ(server, qubits, stats, raw=True)])
        prob = mp.tunnelingv1([q], data)[0]
        if noisy:
            print np.round([k,m,prob],3)
            
        returnValue([prob])
    
    funcWrapper = popt.makeFuncWrapper(k, func, axes, deps, measure, sample)
    output = sweeps.fmin(funcWrapper, inputs, 
                         dataset, xtol = xtol, ftol = ftol, nonzdelt = nonzdelt, 
                         zdelt = zdelt, maxfun=maxfun)
    
    return output
    
def orbit_1q_v2(sample, m, k = 40, alpha=None, piAmp=None, piAmpHalf=None, df=None, gate='X', tbuffer=0*ns, measure=0, measureLevel=1, stats=600, name='orbit 1q v2', save=True, noisy=True, collect=False):
    '''
    optimize pi pulse
    Single Qubit Randomized Benchmarking.
    Parameters(only for target gate):
    m(list):the maximum number of the Cliffords;
    k(integer):the nummber of different random sequence;
    gate(string):the gate to marking;
    df(string):detune
    ''' 
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    if alpha is None:
        alpha = q.alpha
    if piAmp is None:
        piAmp = q.piAmp
    if piAmpHalf is None:
        piAmpHalf = q.piAmpHalf
    
    if df is None:
        df = q.piDf
    axes = [(np.arange(k), 'k'), (m, 'm'), (alpha, 'alpha'), (piAmp, 'piAmp'), (piAmpHalf, 'piAmpHalf'), (df, 'df')]
    
    kw = {'stats':stats,
          'tbuffer':tbuffer,
          'gate':gate,
          'measure':measure,
          
         }
    
    if np.iterable(m):
        M = np.max(m)
    else:
        M = m
    if measureLevel == 1:
        deps = [('Prob', '|0>', ''), ('Prob', '|1>', '')]   #operations for detection
    elif measureLevel == 2:
        deps = [('Prob', '|0>', ''), ('Prob', '|1>', ''), ('Prob', '|2>', '')] 
    else:
        raise Exception('How many levels do you want to measure??')
    randomNumbers=numpy.random.randint(0,24,(k,M))
    # print randomNumbers
    kw['randomNumbers'] = randomNumbers
    dataset = sweeps.prepDataset(sample, name+gate, axes, deps, measure=measure, kw=kw) 

    def func(server, k, m, alpha, piAmp, piAmpHalf, df):    #define the sequence for randomized benchmarking
        for qubit in qubits:
            qubit['adc_start_delay'] = q['adc_start_delay']
            qubit['readout DAC start delay'] = 0*ns 
            qubit['readout fc'] = q['readout fc']
            qubit['fc'] = q['fc']
        q['adc demod frequency'] = q['qnd_readout frequency']- q['readout fc']
        start = 0
        q.xy = env.NOTHING
        q.z = env.NOTHING
        op = np.diag([1.0,1.0])
        for idx in range(m):
            randomNumber = randomNumbers[k,idx]
            # --------clifford gates------
            start, xy, operator = CliffordFor1(q, start, randomNumber)
            q.xy += xy
            op = np.dot(operator,op)
            start += tbuffer
            # --------target gate------
            start, xy, z, operator = targetGateFor1(q, start, gate, alpha, df)
            q.xy += xy
            q.z += z
            op = np.dot(operator,op)
            start += tbuffer
        # --------recovery gates------
        start, xy = recoveryGateFor1(q, start, op)
        q.xy += xy
        # ---------readout------------
        q.rr = eh.QND_readoutPulse_ring(q,start)#readout
        for qubit in qubits:
            qubit['adc filterStart'] = start + q['qnd_ringLen']
            qubit['adc filterStop'] = qubit['adc filterStart'] + q['qnd_readoutLen']
        q['readout'] = True
        data = yield FutureList([runQ(server, qubits, stats, raw=True)])
        if measureLevel == 1:
            prob = mp.tunnelingv1([q], data)[0]
            if noisy:
                print np.round([k,m,prob],3)
            probs = [1.0-prob,prob]
        elif measureLevel == 2:
            probs = mp.tunneling_multilevel(q, data)
            if noisy:
                print k,m
                print np.round(probs,3)
        else:
            raise Exception('How many levels do you want to measure??')
        returnValue(probs)

    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if collect:
        return data

def orbit_1q_cos(sample, m, k = 40, alpha=None, piAmp=None, df=None, gate='X', tbuffer=0*ns, measure=0, measureLevel=1, stats=600, name='orbit 1q cos', save=True, noisy=True, collect=False):
    '''
    optimize pi pulse
    Single Qubit Randomized Benchmarking.
    Parameters:
    m(list):the maximum number of the Cliffords;
    k(integer):the nummber of different random sequence;
    gate(string):the gate to marking;
    ''' 
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    if alpha is None:
        alpha = q.alpha
    if piAmp is None:
        piAmp = q.piAmpCosine
    if df is None:
        df = q.piDf
    axes = [(np.arange(k), 'k'), (m, 'm'), (alpha, 'alpha'), (piAmp, 'piAmp'), (df, 'df')]
    
    kw = {'stats':stats,
          'tbuffer':tbuffer,
          'gate':gate,
          'measure':measure,
          
         }
    
    if np.iterable(m):
        M = np.max(m)
    else:
        M = m
    if measureLevel == 1:
        deps = [('Prob', '|0>', ''), ('Prob', '|1>', '')]   #operations for detection
    elif measureLevel == 2:
        deps = [('Prob', '|0>', ''), ('Prob', '|1>', ''), ('Prob', '|2>', '')] 
    else:
        raise Exception('How many levels do you want to measure??')
    randomNumbers=numpy.random.randint(0,24,(k,M))
    # kw['randomNumbers'] = randomNumbers
    dataset = sweeps.prepDataset(sample, name+gate, axes, deps, measure=measure, kw=kw) 
    
    def func(server, k, m, alpha, piAmp, df):    #define the sequence for randomized benchmarking
        for qubit in qubits:
            qubit['adc_start_delay'] = q['adc_start_delay']
            qubit['readout DAC start delay'] = 0*ns 
            qubit['readout fc'] = q['readout fc']
            qubit['fc'] = q['fc']
        q['adc demod frequency'] = q['qnd_readout frequency']- q['readout fc']
        q.piAmpCosine = piAmp
        q.alpha = alpha
        q.piDf = df
        start = 0
        q.xy = env.NOTHING
        q.z = env.NOTHING
        op = np.diag([1.0,1.0])
        for idx in range(m):
            randomNumber = randomNumbers[k,idx]
            #---------clifford gates------
            start, xy, operator = CliffordFor1_cos(q, start, randomNumber)
            q.xy += xy
            op = np.dot(operator,op)
            start += tbuffer
            #---------target gate------
            start, xy, z, operator = targetGateFor1(q, start, gate)
            q.xy += xy
            q.z += z
            op = np.dot(operator,op)
            start += tbuffer
        #---------recovery gates------
        start, xy = recoveryGateFor1(q, start, op)
        q.xy += xy
        #----------readout------------
        q.rr = eh.QND_readoutPulse_ring(q,start)#readout
        for qubit in qubits:
            qubit['adc filterStart'] = start + q['qnd_ringLen']
            qubit['adc filterStop'] = qubit['adc filterStart'] + q['qnd_readoutLen']
        q['readout'] = True
        data = yield FutureList([runQ(server, qubits, stats, raw=True)])
        if measureLevel == 1:
            prob = mp.tunnelingv1([q], data)[0]
            if noisy:
                print np.round([k,m,prob],3)
            probs = [1.0-prob,prob]
        elif measureLevel == 2:
            probs = mp.tunneling_multilevel(q, data)
            if noisy:
                print k,m
                print np.round(probs,3)
        else:
            raise Exception('How many levels do you want to measure??')
        returnValue(probs)

    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if collect:
        return data        
        
def randomizedBenchmarking_2q(sample, m, k = 40, tbuffer=0*ns, measure=[2,0,3], stats=600, gate='CZ',
                              Omega2=7.0, delay=250*ns, delta=4*MHz, zpa_comp1s=None, zpa_comp2s=None, dir=None, filename = None, des='',
                              name='Randomized Benchmarking 2q ', save=True, noisy=True, collect=False):
    '''
    Two Qubits Randomized Benchmarking.
    Load data for random sequences from outside file.
    Actually three 'qubits' are used here: r, q1, and q2; r is used in geometric phase gate
    Parameters:
    m(list):the maximum number of the Cliffords;
    k(integer):the nummber of different random sequence;
    ''' 
    if dir is None:
        dir = 'D:\\songchao\\RBsequence\\'
    if filename is None:
        filename = 'op_list_ref'
    result = scipy.io.loadmat(dir+filename+'.mat')
    RBdata = result['op_list']
    sample, qubits = util.loadQubits(sample)
    qubits = np.array(qubits)
    r, qA, qB = qubits[measure]
    
    if zpa_comp1s is None:
        zpa_comp1s = qA.zpa_comp
    if zpa_comp2s is None:
        zpa_comp2s = qB.zpa_comp
    axes = [(np.arange(k), 'k'), (m, 'm'),(tbuffer,'tbuffer'),(Omega2, 'Omega2'),(zpa_comp1s,'zpa_comp1s'),(zpa_comp2s,'zpa_comp2s')]
    
    kw = {'stats':stats,
          'measure':measure,
          'delay':delay,
          'delta':delta,
          'gate':gate,
         }
    
    if np.iterable(m):
        M = np.max(m)
    else:
        M = m
    deps = []
    lbs = ['0','1']
    for i in lbs:
        for j in lbs:
            deps.append(('Prob', '|'+i+j+'>', ''))
    randomNumbers0 = RBdata[:k+1,:M+1,0,0]
    randomNumbers1 = RBdata[:k+1,:M+1,0,1]
    recoveryGateInfo = RBdata[:k+1,:M+1,1:,:]
    dataset = sweeps.prepDataset(sample, name+gate+des, axes, deps, measure=measure, kw=kw) 
    
    def func(server, k, m, tbuffer, Omega2, zpaA, zpaB):    #define the sequence for randomized benchmarking
        for qubit in qubits:
            qubit['adc_start_delay'] = qB['adc_start_delay']
            qubit['readout DAC start delay'] = 0*ns 
            qubit['readout fc'] = qB['readout fc']
            qubit['fc'] = qB['fc']
        for qubit in qubits[measure]:
            qubit['adc demod frequency'] = qubit['qnd_readout frequency'] - qubit['readout fc']
            qubit.xy = env.NOTHING
            qubit.z = env.NOTHING
        qA.zpa_comp = zpaA
        qB.zpa_comp = zpaB
        print qA.zpa_comp, qB.zpa_comp
        piLen = np.max([qubit.piLen for qubit in [qA, qB]])
        start = 0*ns    #initialization
        op = np.diag([1.0,1.0])
        ops = [op for i in range(2)]
        op = reduce(np.kron, ops)
        for idxm in range(m):
            randomNumber0 = randomNumbers0[k, idxm+1]
            randomNumber1 = randomNumbers1[k, idxm+1]
            #---------clifford gates------
            start, xys, operator = CliffordForN([qA, qB], start, [randomNumber0, randomNumber1])
            op = np.dot(operator, op)
            for idx, qubit in enumerate([qA, qB]):
                qubit.xy += xys[idx]
            start += tbuffer
            #---------target gate------
            if gate == 'CZ':
                start, xys, zs, operator = geomCZGate(r, [qA, qB], start, Omega2, delay, delta)
                for idx, qubit in enumerate(qubits[measure]):
                    qubit.xy += xys[idx]
                    qubit.z += zs[idx]
            else:
                if gate == 'reference':
                    gates = [gate, gate]
                else:
                    gates = gate
                start, xys, operator = targetGateForN([qA, qB], start, gates)
                for idx, qubit in enumerate([qA, qB]):
                    qubit.xy += xys[idx]
            op = np.dot(operator, op)
            start += tbuffer
        # print 'debug1',start
        #---------recovery gate------  
        start, xys, zs = recoveryGateFor2(r, [qA,qB], start, recoveryGateInfo[k,m,:,:], Omega2, delay, delta, op, False, tbuffer)
        start = start*ns
        for idx, qubit in enumerate(qubits[measure]):
            qubit.xy += xys[idx]
            qubit.z += zs[idx]
        #---------readout------------  
        for qubit in [qA, qB]:
            qubit.rr = eh.QND_readoutPulse_ring(qubit,start)#readout
            qubit['readout'] = True
        for qubit in qubits:
            qubit['adc filterStart'] = start + qB['qnd_ringLen']
            # print 'debug 2: ', start, qubit['adc filterStart']
            qubit['adc filterStop'] = qubit['adc filterStart'] + qB['qnd_readoutLen']
        data = yield FutureList([runQ(server, qubits, stats, raw=True)])
        prob = mp.tunnelingv2([qA, qB], data)
        if noisy:
            print 'k: %d, m: %d'%(k, m)
            print np.round(prob,3)
        returnValue(prob)

    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if collect:
        return data 
        
def randomizedBenchmarking_2q_v1(sample, m, k = 40, tbuffer=0*ns, measure=[2,0,3], stats=600, gate='CZ',
                              Omega2=7.0, delay=250*ns, delta=4*MHz, des='', zpa_comp1s=None, zpa_comp2s=None, 
                              name='Randomized Benchmarking 2q ', save=True, noisy=True, collect=False):
    '''
    Two Qubits Randomized Benchmarking.
    Generate RBdata before exp.
    Actually three 'qubits' are used here: r, q1, and q2; r is used in geometric phase gate
    Parameters:
    m(list):the maximum number of the Cliffords;
    k(integer):the nummber of different random sequence;
    ''' 
    sample, qubits = util.loadQubits(sample)
    qubits = np.array(qubits)
    r, qA, qB = qubits[measure]
    if zpa_comp1s is None:
        zpa_comp1s = qA.zpa_comp
    if zpa_comp2s is None:
        zpa_comp2s = qB.zpa_comp
    axes = [(np.arange(k), 'k'), (m, 'm'),(tbuffer,'tbuffer'),(Omega2,'Omega2'),(zpa_comp1s,'zpa_comp1s'),(zpa_comp2s,'zpa_comp2s'),(delta,'delta')]
    
    kw = {'stats':stats,
          'measure':measure,
          'delay':delay,
          'gate':gate,
         }
    
    if np.iterable(m):
        M = np.max(m)
    else:
        M = m
    deps = []
    lbs = ['0','1']
    for i in lbs:
        for j in lbs:
            deps.append(('Prob', '|'+i+j+'>', ''))
    
    if gate == 'CZ':
        RBdata = generateSeq_cz.generateSeq_cz(k=k, m=M)
    else:
        if gate == 'reference':gates = [gate, gate]
        else: gates = gate
        RBdata = generateSeq_gate.generateSeq_gate(k=k, m=M, gates=gates)
    
    randomNumbers0 = RBdata[:,:,0,0]
    randomNumbers1 = RBdata[:,:,0,1]
    recoveryGateInfo = RBdata[:,:,1:,:]
    dataset = sweeps.prepDataset(sample, name+gate+des, axes, deps, measure=measure, kw=kw) 
    
    def func(server, k, m, tbuffer, Omega2, zpaA, zpaB, delta):    #define the sequence for randomized benchmarking
        for qubit in qubits:
            qubit['adc_start_delay'] = qB['adc_start_delay']
            qubit['readout DAC start delay'] = 0*ns 
            qubit['readout fc'] = qB['readout fc']
            qubit['fc'] = qB['fc']
        for qubit in qubits[measure]:
            qubit['adc demod frequency'] = qubit['qnd_readout frequency'] - qubit['readout fc']
            qubit.xy = env.NOTHING
            qubit.z = env.NOTHING
        qA.zpa_comp = zpaA
        qB.zpa_comp = zpaB
        piLen = np.max([qubit.piLen for qubit in [qA, qB]])
        start = 0*ns    #initialization
        op = np.diag([1.0,1.0])
        ops = [op for i in range(2)]
        op = reduce(np.kron, ops)
        for idxm in range(m):
            randomNumber0 = randomNumbers0[k, idxm+1]
            randomNumber1 = randomNumbers1[k, idxm+1]
            #---------clifford gates------
            start, xys, operator = CliffordForN([qA, qB], start, [randomNumber0, randomNumber1])
            op = np.dot(operator, op)
            for idx, qubit in enumerate([qA, qB]):
                qubit.xy += xys[idx]
            start += tbuffer
            #---------target gate------
            if gate == 'CZ':
                start, xys, zs, operator = geomCZGate(r, [qA, qB], start, Omega2, delay, delta)
                for idx, qubit in enumerate(qubits[measure]):
                    qubit.xy += xys[idx]
                    qubit.z += zs[idx]
            else:
                if gate == 'reference':
                    gates = [gate, gate]
                else:
                    gates = gate
                start, xys, operator = targetGateForN([qA, qB], start, gates)
                for idx, qubit in enumerate([qA, qB]):
                    qubit.xy += xys[idx]
            op = np.dot(operator, op)
            start += tbuffer
        # print 'debug1',start
        #---------recovery gate------  
        start, xys, zs = recoveryGateFor2(r, [qA,qB], start, recoveryGateInfo[k,m,:,:], Omega2, delay, delta, op, False, tbuffer)
        start = start*ns
        for idx, qubit in enumerate(qubits[measure]):
            qubit.xy += xys[idx]
            qubit.z += zs[idx]
        #---------readout------------  
        for qubit in [qA, qB]:
            qubit.rr = eh.QND_readoutPulse_ring(qubit,start)#readout
            qubit['readout'] = True
        for qubit in qubits:
            qubit['adc filterStart'] = start + qB['qnd_ringLen']
            # print 'debug 2: ', start, qubit['adc filterStart']
            qubit['adc filterStop'] = qubit['adc filterStart'] + qB['qnd_readoutLen']
        data = yield FutureList([runQ(server, qubits, stats, raw=True)])
        prob = mp.tunnelingv2([qA, qB], data)
        if noisy:
            print 'k: %d, m: %d'%(k, m)
            print np.round(prob,3)
        returnValue(prob)

    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if collect:
        return data         
        
def randomizedBenchmarking_2q_full(sample, m, k = 40, tbuffer=0*ns, measure=[2,0,3], stats=600, gate='CZ',
                              Omega2=7.0, delay=250*ns, delta=4*MHz, des='', zpa_comp1s=None, zpa_comp2s=None, 
                              name='Randomized Benchmarking 2q full', save=True, noisy=True, collect=False):
    '''
    Two Qubits Randomized Benchmarking using full clifford group
    Generate RBdata before exp.
    Actually three 'qubits' are used here: r, q1, and q2; r is used in geometric phase gate
    Parameters:
    m(list):the maximum number of the Cliffords;
    k(integer):the nummber of different random sequence;
    ''' 
    sample, qubits = util.loadQubits(sample)
    qubits = np.array(qubits)
    r, qA, qB = qubits[measure]
    if zpa_comp1s is None:
        zpa_comp1s = qA.zpa_comp
    if zpa_comp2s is None:
        zpa_comp2s = qB.zpa_comp
    axes = [(np.arange(k), 'k'), (m, 'm'),(tbuffer,'tbuffer'),(Omega2,'Omega2'),(zpa_comp1s,'zpa_comp1s'),(zpa_comp2s,'zpa_comp2s'),(delta,'delta')]
    
    kw = {'stats':stats,
          'measure':measure,
          'delay':delay,
          'gate':gate,
         }
    
    if np.iterable(m):
        M = np.max(m)
    else:
        M = m
    deps = []
    lbs = ['0','1']
    for i in lbs:
        for j in lbs:
            deps.append(('Prob', '|'+i+j+'>', ''))

    RBdata = generateSeq_gate_full.generateSeq_gate_full(k=k, m=M, gate=gate)
    
    randomNumbers = RBdata[:,:,:7,:]
    recoveryGateInfo = RBdata[:,:,7:,:]
    dataset = sweeps.prepDataset(sample, name+gate+des, axes, deps, measure=measure, kw=kw) 
    
    def func(server, k, m, tbuffer, Omega2, zpaA, zpaB, delta):    #define the sequence for randomized benchmarking
        for qubit in qubits:
            qubit['adc_start_delay'] = qB['adc_start_delay']
            qubit['readout DAC start delay'] = 0*ns 
            qubit['readout fc'] = qB['readout fc']
            qubit['fc'] = qB['fc']
        for qubit in qubits[measure]:
            qubit['adc demod frequency'] = qubit['qnd_readout frequency'] - qubit['readout fc']
            qubit.xy = env.NOTHING
            qubit.z = env.NOTHING
        qA.zpa_comp = zpaA
        qB.zpa_comp = zpaB
        piLen = np.max([qubit.piLen for qubit in [qA, qB]])
        start = 0*ns    #initialization
        op = np.diag([1.0,1.0])
        ops = [op for i in range(2)]
        op = reduce(np.kron, ops)
        for idxm in range(m):
            #---------clifford gates------
            start, xys, zs, operator = fullCliffordFor2(r, [qA, qB], start, randomNumbers[k, idxm+1,:,:], Omega2, delay, delta)
            op = np.dot(operator, op)
            for idx, qubit in enumerate(qubits[measure]):
                qubit.xy += xys[idx]
                qubit.z += zs[idx]
            start += tbuffer
            #---------target gate------
            if gate == 'CZ':
                start, xys, zs, operator = geomCZGate(r, [qA, qB], start, Omega2, delay, delta)
                for idx, qubit in enumerate(qubits[measure]):
                    qubit.xy += xys[idx]
                    qubit.z += zs[idx]
            else:
                if gate == 'reference':
                    gates = [gate, gate]
                else:
                    gates = gate
                start, xys, operator = targetGateForN([qA, qB], start, gates)
                for idx, qubit in enumerate([qA, qB]):
                    qubit.xy += xys[idx]
            op = np.dot(operator, op)
            start += tbuffer
        # print 'debug1',start
        #---------recovery gate------  
        start, xys, zs, operator = fullCliffordFor2(r, [qA,qB], start, recoveryGateInfo[k,m,:,:], Omega2, delay, delta)
        start = start*ns
        for idx, qubit in enumerate(qubits[measure]):
            qubit.xy += xys[idx]
            qubit.z += zs[idx]
        #---------readout------------  
        for qubit in [qA, qB]:
            qubit.rr = eh.QND_readoutPulse_ring(qubit,start)#readout
            qubit['readout'] = True
        for qubit in qubits:
            qubit['adc filterStart'] = start + qB['qnd_ringLen']
            # print 'debug 2: ', start, qubit['adc filterStart']
            qubit['adc filterStop'] = qubit['adc filterStart'] + qB['qnd_readoutLen']
        data = yield FutureList([runQ(server, qubits, stats, raw=True)])
        prob = mp.tunnelingv2([qA, qB], data)
        if noisy:
            print 'k: %d, m: %d'%(k, m)
            print np.round(prob,3)
        returnValue(prob)

    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if collect:
        return data     
        
def randomizedBenchmarking_3q(sample, m, k = 40, tbuffer=0*ns, measure=[2,0,3], stats=600, gate='CZ',
                              Omega2=7.0, delay=250*ns, delta=4*MHz, des='', zpa_comp1s=None, zpa_comp2s=None, zpa_comp3s=None, 
                              name='Randomized Benchmarking 3q', save=True, noisy=True, collect=False):
    '''
    Three Qubits Randomized Benchmarking using Pauli-based clifford group
    load RBdata before exp.
    Actually four 'qubits' are used here: r, q1, q2 and q3; r is used in geometric phase gate
    Parameters:
    m(list):the maximum number of the Cliffords;
    k(integer):the nummber of different random sequence;
    ''' 
    sample, qubits = util.loadQubits(sample)
    qubits = np.array(qubits)
    r, qA, qB, qC = qubits[measure]
    if zpa_comp1s is None:
        zpa_comp1s = qA.zpa_comp
    if zpa_comp2s is None:
        zpa_comp2s = qB.zpa_comp
    if zpa_comp3s is None:
        zpa_comp3s = qC.zpa_comp
    
    axes = [(np.arange(k), 'k'), (m, 'm'),(tbuffer,'tbuffer'),(Omega2,'Omega2'),(zpa_comp1s,'zpa_comp1s'),(zpa_comp2s,'zpa_comp2s'),(delta,'delta')]
    
    kw = {'stats':stats,
          'measure':measure,
          'delay':delay,
          'gate':gate,
         }
    
    if np.iterable(m):
        M = np.max(m)
    else:
        M = m
    deps = []
    lbs = ['0','1']
    for i in lbs:
        for j in lbs:
            for k in lbs:
            
                deps.append(('Prob', '|'+i+j+k+'>', ''))

    RBdata = generateSeq_gate_full.generateSeq_gate_full(k=k, m=M, gate=gate)
    
    randomNumbers = RBdata[:,:,:7,:]
    recoveryGateInfo = RBdata[:,:,7:,:]
    dataset = sweeps.prepDataset(sample, name+gate+des, axes, deps, measure=measure, kw=kw) 
    
    def func(server, k, m, tbuffer, Omega2, zpaA, zpaB, zpaC, delta):    #define the sequence for randomized benchmarking
        for qubit in qubits:
            qubit['adc_start_delay'] = qB['adc_start_delay']
            qubit['readout DAC start delay'] = 0*ns 
            qubit['readout fc'] = qB['readout fc']
            qubit['fc'] = qB['fc']
        for qubit in qubits[measure]:
            qubit['adc demod frequency'] = qubit['qnd_readout frequency'] - qubit['readout fc']
            qubit.xy = env.NOTHING
            qubit.z = env.NOTHING
        qA.zpa_comp = zpaA
        qB.zpa_comp = zpaB
        qC.zpa_comp = zpaC
        piLen = np.max([qubit.piLen for qubit in [qA, qB, qC]])
        start = 0*ns    #initialization
        op = np.diag([1.0,1.0])
        ops = [op for i in range(2)]
        op = reduce(np.kron, ops)
        for idxm in range(m):
            #---------clifford gates------
            start, xys, zs, operator = fullCliffordFor2(r, [qA, qB, qC], start, randomNumbers[k, idxm+1,:,:], Omega2, delay, delta)
            op = np.dot(operator, op)
            for idx, qubit in enumerate(qubits[measure]):
                qubit.xy += xys[idx]
                qubit.z += zs[idx]
            start += tbuffer
            #---------target gate------
            if gate == 'CZ':
                start, xys, zs, operator = geomCZGate(r, [qA, qB, qC], start, Omega2, delay, delta)
                for idx, qubit in enumerate(qubits[measure]):
                    qubit.xy += xys[idx]
                    qubit.z += zs[idx]
            else:
                if gate == 'reference':
                    gates = [gate, gate]
                else:
                    gates = gate
                start, xys, operator = targetGateForN([qA, qB, qC], start, gates)
                for idx, qubit in enumerate([qA, qB, qC]):
                    qubit.xy += xys[idx]
            op = np.dot(operator, op)
            start += tbuffer
        # print 'debug1',start
        #---------recovery gate------  
        start, xys, zs, operator = fullCliffordFor2(r, [qA,qB, qC], start, recoveryGateInfo[k,m,:,:], Omega2, delay, delta)
        start = start*ns
        for idx, qubit in enumerate(qubits[measure]):
            qubit.xy += xys[idx]
            qubit.z += zs[idx]
        #---------readout------------  
        for qubit in [qA, qB, qC]:
            qubit.rr = eh.QND_readoutPulse_ring(qubit,start)#readout
            qubit['readout'] = True
        for qubit in qubits:
            qubit['adc filterStart'] = start + qB['qnd_ringLen']
            # print 'debug 2: ', start, qubit['adc filterStart']
            qubit['adc filterStop'] = qubit['adc filterStart'] + qB['qnd_readoutLen']
        data = yield FutureList([runQ(server, qubits, stats, raw=True)])
        prob = mp.tunnelingv3([qA, qB, qC], data).flatten()
        if noisy:
            print 'k: %d, m: %d'%(k, m)
            print np.round(prob,3)
        returnValue(prob)

    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if collect:
        return data 
        
def overmorning20170412(s):
    orbit_1q(s, k=20, gate='reference', measure=0)     
    orbit_1q(s, k=20, gate='X', measure=0)   
    orbit_1q(s, k=20, gate='X/2', measure=0) 
    
    orbit_1q(s, k=20, gate='reference', measure=1)     
    orbit_1q(s, k=20, gate='X', measure=1)   
    orbit_1q(s, k=20, gate='X/2', measure=1)   
    
    # mp.T1_visibility(s, delay=st.r[0:14:0.05, us], measure=0, measure_xy=0, stats=600)
    # mp.ramsey(s, delay=st.r[0:14:0.01,us], measure=0, measure_xy=0, stats=600, fringeFreq=2*MHz)
   