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