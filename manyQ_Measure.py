def tunnelingN_peach(qubits, data):
	# peach ------ 190506
    qNum = len(qubits)
    counts_num = len(np.asarray(data[0][0][0]))
    binary_count = np.zeros((counts_num),dtype = float)
    #eg:   q2 =1 q1 = 0 ---->  1 0    ---> 2
    #eg:   q2 =1 q1 = 1 ---->  1 1    ---> 3
    def get_meas(data0,q):
        #data[0][x]   qubits[x]   x == 0,1,2,3
        # if measure 1 then return 1
        Is1 = np.asarray(data0[0])
        Qs1 = np.asarray(data0[1])    
        sigs1 = Is1 + 1j*Qs1
        center_0 = q['center|0>'][0] + 1j*q['center|0>'][1]
        center_1 = q['center|1>'][0] + 1j*q['center|1>'][1]
        distance_0 = np.abs(sigs1-center_0)
        distance_1 = np.abs(sigs1-center_1)
        meas1 = np.asarray(distance_0 > distance_1,dtype = float)
        return meas1
    for i in np.arange(qNum):
        binary_count += get_meas(data[0][i],qubits[i]) * (2**i)
    res_store = np.zeros((2**qNum))
    for i in np.arange(2**qNum):
        res_store[i] = np.sum(binary_count == i) 
    prob = res_store/counts_num
    return prob


def tunnelingNlevelQ_peach(qubits, data,level = 3,qNum = 1):
    ## generated to N qubit and multi level 20190618 -- ZiyuTao
    qNum = len(qubits)
    counts_num = len(np.asarray(data[0][0][0]))
    binary_count = np.zeros((counts_num),dtype = float)

    def get_meas(data0,q,Nq = level):
        #data[0][x]   qubits[x]   x == 0,1,2,3
        # if measure 1 then return 1
        Is = np.asarray(data0[0])
        Qs = np.asarray(data0[1])    
        sigs = Is + 1j*Qs
        
        total = len(Is)
        distance = np.zeros((total,Nq))
        for i in np.arange(Nq):
            center_i = q['center|'+str(i)+'>'][0] + 1j*q['center|'+str(i)+'>'][1]
            distance_i = np.abs(sigs - center_i)
            distance[:,i]=  distance_i
        
        tunnels = np.zeros((total,))
        for i in np.arange(total):
            distancei = distance[i]
            tunneli = np.int(np.where(distancei == np.min(distancei))[0])
            tunnels[i] = tunneli 
        return tunnels

    for i in np.arange(qNum):
        binary_count += get_meas(data[0][i],qubits[i]) * (level**i)
        

    res_store = np.zeros((level**qNum))
    for i in np.arange(level**qNum):
        res_store[i] = np.sum(binary_count == i) 
        
    prob = res_store/counts_num
    return prob
