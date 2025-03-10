# Test Data Folder Structure

This folder contains various subdirectories that store `.stim` files, which are quantum error correction circuit descriptions used for testing and benchmarking. The `.stim` files follow a specific naming convention, encoding parameters like distance (`d`), rounds (`r`), noise type (`noise`), and gate set (`gates`).

## Folder Structure

```
 testdata/
 ├── surfacecodes/
 ├── colorcodes/
 ├── bivariatebicyclecodes/
 ├── surface_code_trans_cx_circuits/
```

### 1. `surfacecodes/`
This folder contains `.stim` files for rotated surface codes. These protocols were introduced in:
Eric Dennis, Alexei Kitaev, Andrew Landahl, John Preskill; Topological quantum memory. J. Math. Phys. 1 September 2002; 43 (9): 4452–4505. https://doi.org/10.1063/1.1499754

The filenames encode parameters such as:
- `r`: Number of rounds
- `d`: Code distance
- `p`: Noise probability
- `noise`: Noise model used (e.g., `si1000`)
- `c`: Code type (e.g., `surface_code_X`)
- `q`: Number of qubits
- `gates`: Gates used in the circuit (e.g., `cz` for controlled-Z gates)

Example filename:
```
r=11,d=11,p=0.0001,noise=si1000,c=surface_code_X,q=241,gates=cz.stim
```

### 2. `colorcodes/`
Contains `.stim` files for color code memory protocols. The specific syndrome extraction circuit is the 'superdense' cycle from:
Gidney, Craig, and Cody Jones. "New circuits and an open source decoder for the color code." arXiv preprint arXiv:2312.08813 (2023).
The filenames encode parameters such as:
- `r`: Number of rounds
- `d`: Code distance
- `p`: Noise probability
- `noise`: Noise model used (e.g., `data_qubit_X`)
- `c`: Code type (e.g., `color_code_mpp`)
- `q`: Number of qubits
- `gates`: Gates used in the circuit (e.g., `mpp` for measurement-based parity check gates)

Example filename:
```
r=1,d=11,p=0.01,noise=data_qubit_X,c=color_code_mpp,q=91,gates=mpp.stim
```

### 3. `bivariatebicyclecodes/`
Contains `.stim` files for bivariate bicycle codes, as studied in:
Bravyi, S., Cross, A.W., Gambetta, J.M. et al. High-threshold and low-overhead fault-tolerant quantum memory. Nature 627, 778–782 (2024). https://doi.org/10.1038/s41586-024-07107-7

These files include additional parameters related to the code structure, including:
- `r`: Number of rounds
- `d`: Code distance
- `p`: Noise probability
- `noise`: Noise model used (e.g., `si1000`)
- `c`: Code type (e.g., `bivariate_bicycle_X`)
- `nkd`: Nested code parameters, typically in matrix format
- `q`: Number of qubits
- `iscolored`: Boolean flag indicating whether the code has a color structure
- `A_poly`: Polynomial descriptor for matrix A
- `B_poly`: Polynomial descriptor for matrix B

Example filename:
```
r=10,d=10,p=0.0001,noise=si1000,c=bivariate_bicycle_X,nkd=[[108,8,10]],q=216,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim
```

### 4. `surface_code_trans_cx_circuits/`
This folder contains `.stim` files for surface code circuits that implement transversal CX (CNOT) gates. These protocols are naturally suited for neutral atom architectures and the correlated decoding of their error models was studied in:
Cain, Madelyn, et al. "Correlated decoding of logical algorithms with transversal gates." Physical Review Letters 133.24 (2024): 240602.

The `.stim` file naming follows the same convention as the other directories but may include CX-specific parameters.

Example filename:
```
r=11,d=11,p=0.0001,noise=si1000,c=surface_code_trans_cx_X,q=482,gates=cz.stim
```