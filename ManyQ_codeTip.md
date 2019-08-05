
## Example: N qubit state tomography

We need $3^N$ operations $\{I,X_{\pi/2},Y_{\pi/2}\}^{\otimes N}$ which generally acting on N qubits $(q_1,q_2,q_3 \cdots)$, so we need two loops.

```python
#!/usr/bin/env python3
for tomo_idx in range(3**N):
    for q_idx in range(N):
        q_tomo = qlist[N-1-q_idx]
        gate_idx = (tomo_idx//(3**(q_idx+1)))%(3**(q_idx)
        q_tomo.add_gate(gates[gate_idx])
```

### Mathtips: 
it transforms tomo_idx into **3-digit** where the n-th digit represents an operation, e.g. "0212" correspond 
    to the operations $I \otimes Y/2 \otimes X/2 \otimes Y/2$.

### Others: 
- qlist[N-1-q_idx] is set as we prefer denote qlist as (q1,q2,q3) and the digit system is from right to left.

- the quantum operation usually definied by a two-tuple, ($\theta$,$\phi$), so we can define as following

```python
thetas = [0,np.pi/2,np.pi/2]
phis = [0,0,np.pi/2]
## gate_idx == 0,1,2
gate_tomo = get_gate(thetas[gate_idx],phis[gate_idx])
```


```python

```
