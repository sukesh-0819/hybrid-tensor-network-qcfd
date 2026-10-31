# End-to-End Mathematical Description

This note follows one time step from the Burgers equation to the variational circuit and then to the tensor-network implementation. It describes the equations implemented by this repository, rather than assuming a quantum computer solves the PDE automatically.

## 1. Problem and notation

The target is the one-dimensional viscous Burgers equation

```math
\frac{\partial u}{\partial t}
+ u\frac{\partial u}{\partial x}
= \nu\frac{\partial^2 u}{\partial x^2},
\qquad 0 \le x \le L,
```

with zero boundary values. The code initializes the grid using $u_i^0=\sin(\pi x_i)$, then fixes the two endpoints to zero. With the demo's L=1, this is the standard $\sin(\pi x/L)$ profile.

The code uses N equally spaced grid points, viscosity $\nu$, and final time T. Its default demonstration uses N=8, L=1, $\nu=0.1$, and T=1. The objective is to advance a vector of grid values, not to solve a continuum equation exactly.

Let $u^j\in\mathbb{R}^N$ be the discrete field at time $t_j=j\Delta t$. The superscript j denotes a time index.

## 2. Finite-difference reference update

The grid spacing is

```math
\Delta x = \frac{L}{N-1}, \qquad x_i=i\Delta x,\quad i=0,\dots,N-1.
```

At interior points, the centered first and second derivatives are

```math
(D_1 u)_i = \frac{u_{i+1}-u_{i-1}}{2\Delta x},
```

```math
(D_2 u)_i = \frac{u_{i-1}-2u_i+u_{i+1}}{\Delta x^2}.
```

The code stores these as N-by-N matrices. Boundary rows are zero and the two endpoint values are reset to zero after every step. A forward-Euler step gives

```math
u^{j+1}
= u^j + \Delta t\left(\nu D_2u^j
- \mathrm{diag}(u^j)D_1u^j\right).
\tag{1}
```

Here $\mathrm{diag}(u)$ is a diagonal matrix whose diagonal entries are the values of u. It makes the nonlinear product $u\,u_x$ a matrix-vector product for a fixed current field.

The code chooses a conservative step size using

```math
\Delta t_{\max}=0.4\min\left(\frac{\Delta x^2}{\nu},
\frac{\Delta x}{u_{\max}}\right),
```

then chooses a number of steps that is a multiple of four so that the saved comparison times are step boundaries. For the defaults this yields 20 steps and $\Delta t=0.05$. This is the implementation's step-size heuristic for the demo.

## 3. Amplitude representation of the field

A quantum state must have unit length. The code therefore separates the field's direction from its magnitude:

```math
r_j = \|u^j\|_2,\qquad
\psi_j = \frac{u^j}{r_j},\qquad
u^j=r_j\psi_j,\qquad
\|\psi_j\|_2=1.
\tag{2}
```

The normalized vector $\psi_j$ is represented by amplitudes of n data qubits, where

```math
N=2^n,\qquad
|\psi_j\rangle=\sum_{i=0}^{N-1}(\psi_j)_i |i\rangle.
```

The scalar $r_j$ is tracked classically. With N=8, three data qubits provide eight basis states, one for each grid index. The grid value is an amplitude associated with a basis index; it is not stored in a separate qubit.

Substitute $u^j=r_j\psi_j$ into (1). Because the derivative is linear,

```math
u^{j+1}
= r_j\left[
I+\Delta t\,\nu D_2
-\Delta t\,r_j\mathrm{diag}(\psi_j)D_1
\right]\psi_j.
\tag{3}
```

Define

```math
O_L=I+\Delta t\,\nu D_2,\qquad
O_C=-\Delta t\,D_1,
```

```math
D_{\psi_j}=\mathrm{diag}(\psi_j),\qquad
M_j=O_L+r_jD_{\psi_j}O_C.
\tag{4}
```

Then the update is simply

```math
u^{j+1}=r_jM_j\psi_j,\quad
r_{j+1}=r_j\|M_j\psi_j\|_2,\quad
\psi_{j+1}=\frac{M_j\psi_j}{\|M_j\psi_j\|_2}.
\tag{5}
```

The key dependency is visible in (4): $O_L$ and $O_C$ are fixed for the run, while $D_{\psi_j}$ changes with the field at every time step.

## 4. Variational state preparation

The ansatz circuit $U(\theta)$ begins with all data qubits in $|0\rangle$. It applies a trainable $R_y(\theta)$ rotation to each data qubit, followed by a chain of CNOT gates between neighboring qubits. Repeating this rotation-and-entanglement pattern gives

```math
|\psi(\theta)\rangle=U(\theta)|0\rangle^{\otimes n}.
```

The rotation matrix used in the tensor-network implementation is

```math
R_y(\theta)=
\begin{bmatrix}
\cos(\theta/2)&-\sin(\theta/2)\\
\sin(\theta/2)&\cos(\theta/2)
\end{bmatrix}.
```

The ansatz is real-valued because it uses only RY and CX gates. For N=8 the script chooses five entangling layers, with six rounds of RY rotations, giving 18 trainable angles and 10 CNOTs.

Before time evolution, a classical optimizer finds angles whose circuit state matches the normalized initial vector. The cost is

```math
C_0(\theta)=1-(\psi_0)^\mathsf{T}\psi(\theta).
```

Several seeded starting points are tried. The signed inner product is used so that a state with the wrong overall sign is not treated as the desired initial state.

At later steps, the same ansatz represents the candidate $\psi_{j+1}$. The classical optimizer updates its angles; the quantum circuit supplies the overlap signal.

## 5. Block encoding each matrix

A block encoding puts a scaled matrix in one block of a larger unitary. For a matrix M, choose

```math
c_M=(1+10^{-6})\|M\|_2,\qquad A=M/c_M,
```

so that A is a contraction. The code constructs the unitary dilation

```math
U_M=
\begin{bmatrix}
A & \sqrt{I-AA^\dagger}\\
\sqrt{I-A^\dagger A} & -A^\dagger
\end{bmatrix}.
\tag{6}
```

The square roots are positive-semidefinite matrix square roots. For an ancilla in state $|0\rangle$, the selected block obeys

```math
(\langle 0|\otimes I)\,U_M\,(|0\rangle\otimes I)=M/c_M.
\tag{7}
```

Thus, after projecting the flag ancilla back onto zero, the data register has undergone the desired matrix divided by its known scale. The scale is carried separately in classical arithmetic.

The run builds:

- $U_L$ for $O_L$, with scale $c_L$.
- $U_C$ for $O_C$, with scale $c_C$.
- In the dense version, $U_{\phi,j}$ for $D_{\psi_j}$, with scale $c_{D,j}$.

For the dense diagonal matrix, its spectral norm is its largest absolute diagonal entry, so $c_{D,j}$ is approximately $\max_i|(\psi_j)_i|$. The small $10^{-6}$ factor keeps the encoded block strictly within the unitary-dilation bound.

The derivative operator includes its minus sign in $O_C=-\Delta t D_1$. The order of the nonlinear product also matters: the circuit applies $U_C$ and then $U_{\phi,j}$, so its selected data block is $D_{\psi_j}O_C$, matching (4).

## 6. Combining the linear and nonlinear terms

The circuit uses an LCU selector qubit, named lcu. The two terms in $M_j$ have scales

```math
\beta_1=c_L,\qquad
\beta_2=c_Cc_{D,j}r_j,\qquad
\beta=\beta_1+\beta_2.
\tag{8}
```

The controlled rotation angle is

```math
\omega=2\arccos\sqrt{\frac{\beta_2}{\beta}}.
\tag{9}
```

Starting from lcu=$|0\rangle$, $R_y(\omega)$ gives amplitude $\sqrt{\beta_2/\beta}$ to selector value 0 (the nonlinear branch) and $\sqrt{\beta_1/\beta}$ to selector value 1 (the linear branch). The circuit applies:

- $U_C$ followed by $U_{\phi,j}$ when lcu is 0.
- $U_L$ when lcu is 1.

It then applies the inverse rotation $R_y(-\omega)$. Projecting lcu back onto zero recombines the two branches with the squared amplitudes as their coefficients. The resulting data block is

```math
\frac{\beta_2}{\beta}\frac{D_{\psi_j}O_C}{c_Cc_{D,j}}
+
\frac{\beta_1}{\beta}\frac{O_L}{c_L}
=
\frac{r_jD_{\psi_j}O_C+O_L}{\beta}
=
\frac{M_j}{\beta}.
\tag{10}
```

This is why the nonlinear scale in (8) includes the current norm $r_j$. The selector weights compensate for the normalization inside each block encoding.

The outer control qubit is put into a superposition by H. One branch is the reference $|0\rangle^{\otimes n}$; the other prepares $\psi_j$ and applies the selected update. After successful projection of the LCU and block-encoding flags, the updated branch contains

```math
\frac{1}{\sqrt{2}}\frac{M_j\psi_j}{\beta}.
\tag{11}
```

The factor $1/\sqrt{2}$ comes from the initial Hadamard.

## 7. Training the next state with an overlap circuit

Let $\theta'$ be a candidate set of next-step angles. The circuit applies the controlled inverse $U^\dagger(\theta')$ to the active branch. Its overlap with the all-zero data state is

```math
\langle 0|^{\otimes n}U^\dagger(\theta')
\frac{M_j|\psi_j\rangle}{\beta}
=
\frac{\langle\psi(\theta')|M_j|\psi_j\rangle}{\beta}.
\tag{12}
```

A final $R_y(-\phi)$ on the outer control converts the reference/update amplitude balance into a control-qubit Z expectation. Probabilities are conditioned on the lcu, A1, and A2 flags all being zero. The optimizer minimizes

```math
C(\theta',\phi)=1-\langle Z\rangle_{\mathrm{postselected}}.
\tag{13}
```

Driving this cost toward zero aligns the candidate state with the normalized update while fitting $\phi$ to the update's magnitude. The exact statevector version evaluates these probabilities directly. If a finite shot count is supplied, the code samples basis outcomes instead.

The code also computes a fidelity from the measured Z expectation $e_z$, $\alpha$, and $\phi$:

```math
F =
\frac{\left[\alpha(e_z+\cos\phi)+e_z-\cos\phi\right]^2}
{4\alpha\sin^2\phi},
\qquad
\alpha=\tan^2(\phi/2).
\tag{14}
```

For diagnostics, it compares this estimate against the direct squared overlap between the trained ansatz state and the normalized classical vector $M_j\psi_j$.

## 8. Recovering the field norm

The scale-adjusted update length is represented by

```math
\alpha_{\mathrm{exact}}=
\frac{\|M_j\psi_j\|_2^2}{\beta^2}.
\tag{15}
```

The circuit extracts $\alpha$ from the trained angle using $\tan^2(\phi/2)$. The code's norm relation is

```math
r_{j+1}
=
\beta\, f_O(\alpha,\phi)\,r_j,
```

where

```math
f_O(\alpha,\phi)=
\frac{1+\alpha}
{2\sin\phi-(\sqrt{\alpha}-1/\sqrt{\alpha})\cos\phi}.
\tag{16}
```

On the intended positive-angle branch with $\alpha=\tan^2(\phi/2)$, $f_O=\sqrt{\alpha}$. Thus (16) gives

```math
r_{j+1}=\beta\sqrt{\alpha}\,r_j
\approx r_j\|M_j\psi_j\|_2,
\tag{17}
```

which is exactly the magnitude relation in (5) when $\alpha$ matches (15). The field returned to the next step is $r_{j+1}\psi(\theta')$.

## 9. Tensor-network field encoding

The dense version forms the full diagonal matrix $D_{\psi_j}$ and its unitary dilation. The tensor-network version replaces that matrix with a matrix-product-state/tensor-train representation of the same amplitude vector.

A length-N state tensor is written as a chain of cores $G^{(k)}$:

```math
(\psi_j)_{i_0\ldots i_{n-1}}
=
\sum_{b_0,\ldots,b_{n-2}}
G^{(0)}_{1,i_0,b_0}
G^{(1)}_{b_0,i_1,b_1}
\cdots
G^{(n-1)}_{b_{n-2},i_{n-1},1}.
\tag{18}
```

The internal indices b are bond indices; their maximum size is the bond dimension $\chi$. The compiler creates cores from the RY/CX ansatz itself. After each CNOT, it reshapes a neighboring pair of cores, performs an SVD, and keeps at most $\chi_{\max}$ singular directions. If no truncation is needed, contracting the cores reproduces the ansatz state.

To encode the diagonal field, each core is converted to a diagonal operator core with physical input/output bits equal. For each site k and bit value i in {0,1}, this gives a local bond matrix $A^{(k,i)}$. A gauge choice redistributes scale along the chain; the code tries each possible canonical center and selects the gauge with the smallest product of local spectral-norm scales.

The bond register stores the internal index, and each site has a flag qubit for its local unitary dilation. The local unitary gate S{k}:{i} is enabled only on the nonlinear branch and when data qubit k equals i. Its target is the bond register plus that site's flag. The sequence of local gates implements the diagonal operator block through the tensor train. The total field scale is the product of the local scales.

For N=8, the default tensor-train bond dimension is two and one bond qubit is sufficient. The full tensor-network circuit has three data qubits, ctrl, lcu, A1, one bond qubit, and three site flags: 10 qubits total. The dense version has 7 total qubits, but stores a dense N-by-N field block. The tensor-network version uses more ancillas in this small example to express the field as local gates. Its potential benefit is structured representation at larger sizes; the saved demo alone does not establish a hardware advantage.

## 10. Complete step and interpretation of the figures

For each time step j, the scripts perform the following sequence:

1. Read the current ansatz angles and classical norm $r_j$.
2. Build the current field encoding: dense $U_{\phi,j}$, or tensor-train local gates.
3. Compute the scales and selector angle $\omega$.
4. Build the controlled update and overlap circuit.
5. Optimize $\theta'$ and $\phi$ from the conditional Z objective.
6. Recover $r_{j+1}$, form $r_{j+1}\psi(\theta')$, and continue.

The circuit figures show these blocks for a single step. The comparison figure shows the discrete classical reference and variational output at several times.

For the supplied N=8 case, the dense comparison plot reports relative L2 errors from roughly $1.2\times10^{-7}$ to $8.0\times10^{-7}$; the tensor-network plot reports roughly $2.3\times10^{-7}$ to $3.4\times10^{-6}$. These are ideal statevector results with exact probabilities in the default configuration. They measure agreement with this small finite-difference reference and ansatz, not performance on noisy hardware or quantum speedup.

The principal implementation constraints are therefore visible in the code: the dense path explicitly constructs matrices and exact statevectors; the tensor-train path has a configurable bond-dimension approximation; and the variational optimizer is classical. These choices make the prototype useful for validating the circuit construction and comparing representations on a small problem.
