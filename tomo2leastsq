## State tomography

### 1. What's the problem

#### 求什么?

未知的一个密度矩阵 $\rho$, 维度为 $N\times N$，考虑每个元素为复数。

#### 已知条件有什么?

通常超导比特里的QST操作为：

- 选择完备的酉算符序列$U=\{U_1,U_2,\cdots U_M\}$，包含$M$个算符，算符与密度矩阵维度相同

  分别将算符作用于未知态，然后进行测量（共$M$次）

- 测量得到$N$组结果，每组结果记作$b$，$b$通常包含 $N$ 个值，就是最后密度矩阵的对角项 $b_i = \left[U_x\rho U_x^{\dagger} \right]_{ii}$



举个栗子，如果测的是三能级系统，那么作用的酉算符就是$3\times 3$的，最后每组结果$b$就是三个态的概率$P_0,P_1,P_2$。

### 2. How to calculate

首先举个栗子，如果测三能级系统$N$次，每次得到$3$个值，最后我们会有$N\times3$的一个矩阵，当然我们也可以直接把它按顺序列下来，看作是一个长度$3N$的“向量”，不管怎样，它的维度是不变的

#### 总结

把要求的量和测量的量都看成向量，那么问题就简单了，这就是一个线性方程组
$$
Ax = b,\qquad \text{calculate $x$ with given $A,b$}
$$
我们要做的就是，已知线性变换$A$，和测量结果$b$，如何求未知的$x$，其中有对应
$$
\{U_1,U_2\cdots U_M\}\to A,\quad \rho \to x,\quad P_{measure} \to b
$$
于是我们只需要把已知的操作和测量结果转换成$A,b$的格式，然后扔给电脑，然后让它解方程求$x$即可，比如 Numpy 里的 np.linalg.lstsq 。



### 3. Mapping 

#### something

- $U=\{U_1,U_2,\cdots U_M \}$，包含$M$个算符

- 量子态为 $N$ 维，密度矩阵为 $N\times N$，（$n$ 个比特对应 $N=2^n$）

- 若将密度矩阵逐行写成向量 $x$，那么向量长度为 $N^2$

  



关于怎么将已知的操作和测量结果映射成 $A,b$ 的格式，首先看最简单的栗子，二能级，用三个算符$U_a,U_b,U_c$做tomo。

三组测量，每次得到$P_0,P_1$，于是我们将$b$写成
$$
b = \left(P_0^a,P_1^a,P_0^b,P_1^b,P_0^c,P_1^c\right)^T
$$
（尽管二能级通常知道$P_0$就有$P_1=1-P_0$，但我们考虑的是更一般的情况）

然后$x$有
$$
x = (\rho_{00},\rho_{01},\rho_{10},\rho_{11})^T
$$
对应的维度：
$$
\text{dim}(A)=(6,4),\quad \text{dim}(x) = (4,1),\quad \text{dim}(b)=(6,1)
$$

##### 关于A

然后我们需要得到$A$，这里不涉及测量的量，完全是理想的理论对应。

这里的话 $b$ 其实只有 $2\times 3 = 6$ 个元素，代表用 $U_i \in \{U_a,U_b,U_c\}$ 操作后测得的概率 $P_j \in \{P_0,P_1\}$，按照程序的计数习惯，我们使用 $K$ 来标记第 $K+1$ 个元素
$$
iN+j = K, \qquad \text{ $i, j$ = divmod($K, N$)}
$$
比如 $K=0$ 代表，用 $U_a$ 测量得到的 $P_0 $。



$b$ 的第 $K+1$ 个元素，$b[K]$，是由 $A$ 的第 $K+1$  行 $A[K,:]$ （参照Numpy的写法）与 $x$ 相乘得到的，同样也等于 $\text{Tr}(\vert j\rangle\langle j \vert U_i\rho U_i^{\dagger})$，其中
$$
\begin{array}{}
\text{Tr}\left(\vert j\rangle\langle j \vert U_i\rho U_i^{\dagger}\right)  
&=
\text{Tr}\left(U_i^{\dagger}\vert j\rangle\langle j \vert U_i \rho \right) \\
&=
\text{Tr}\left[ (U_i[j,:])^{\dagger} U_i[j,:] \rho \right]\\
\end{array}
$$
每个元素展开得到
$$
\sum_{m,n}\left[ (U_i[j,n])^{\dagger} U_i[j,m]\right] \rho[m,n]
$$
它同时也等于
$$
\sum_L A[K,L] x[L]
$$
而我们是将 $\rho$ 这个矩阵逐行逐行的写成一列 $x$ 的，用 $L$ 代表 $x$ 的第 $L+1$ 个元素，这里二能级即为 $L\in \{0,1,2,3\}$，更一般性地我们有它与矩阵的下标 $[n,m]$ 的关系
$$
mN+n = L, \qquad \text{ $m, n$ = divmod($L, N$)}
$$
以此类推，任意的我们都有
$$
A[K,L] = U^i[j,m] \cdot \left( U^i[j,n]\right)^{\dagger}
$$
由此，对于任意的 $U$ 序列我们都可以得到对应的线性变换 $A$，$U_s\equiv \text{Array}([ U_1,U_2\cdots U_M ])\to A$。
$$
A[K,L] = U_s[i,j,m] \cdot \left(U_s[i,j,n] \right)^{\dagger}
$$

##### Coding

```python
A = np.zeros((M*N, N**2), dtype=complex)
for K in range(M*N):
    for L in range(N**2):
        i, j = divmod(K, N)
        m, n = divmod(L, N)                
        A[K, L] = Us[i, j, m] * Us[i, j, n].conj()
```

当密度矩阵维度小的时候，比如$N<16$，相当于四个比特，可以用内存换速度，算的更快

```python
if N <= 16:
    # 1-4 qubits
    def transform(K, L):
        i, j = divmod(K, N)
        m, n = divmod(L, N)
        return Us[i, j, m] * Us[i, j, n].conj()
    U = np.fromfunction(transform, (M*N, N**2), dtype=int)
```





##### 关于 $b$

一般性的 $b$ 为 $M$ 组测量结果，每组测量结果对应 $N$ 个态的概率，$P_n^m$，如果得到的是矩阵格式，可以直接用 Numpy 将它 flatten.

```
b = np.asarray(diags).flatten()
```

$$
\begin{pmatrix}
P_0^0 & P_1^0 & P_2^0 & \cdots & P_{N-1}^0 \\
P_0^1 & P_1^1 & P_2^1 & \cdots & P_{N-1}^1 \\
\vdots &\vdots &\vdots& \ddots& \vdots \\
P_0^{M-1} & P_1^{M-1} & P_2^{M-1} & \cdots & P_{N-1}^{M-1} \\
\end{pmatrix} 
\to \text{Array}([P_0^0,P_1^0,\cdots P_{N-1}^0,P_0^1\cdots P_{N-1}^{M-1} ])
$$



然后我们就有了，一个 $MN \times N^2$ 的线性变换 $A$，和一个 $MN\times 1$ 的列向量 $b$，直接解出 $x$ 即可
$$
A x = b
$$
例如：

```python
rhoFlat, resids, rank, s = np.linalg.lstsq(A,b)
rho = rhoFlat.reshape((N, N))
```







Refer the codes (probably from UCSB) https://github.com/ZeeUTao/QuantumTomography/blob/master/tomo.py


