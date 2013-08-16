//
// aoints.cc
//
// Copyright (C) 2012 Edward Valeev
//
// Author: Edward Valeev <evaleev@vt.edu>
// Maintainer: EV
//
// This file is part of the SC Toolkit.
//
// The SC Toolkit is free software; you can redistribute it and/or modify
// it under the terms of the GNU Library General Public License as published by
// the Free Software Foundation; either version 2, or (at your option)
// any later version.
//
// The SC Toolkit is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Library General Public License for more details.
//
// You should have received a copy of the GNU Library General Public License
// along with the SC Toolkit; see the file COPYING.LIB.  If not, write to
// the Free Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.
//
// The U.S. Government is granted a limited license as per AL 91-7.
//

#include <iostream>
#include <iomanip>
#include <chemistry/molecule/molecule.h>
#include <chemistry/qc/basis/integral.h>
#include <chemistry/qc/basis/split.h>
#include <ctime>
#include <stdlib.h>
#include <stdio.h>

#include <iomanip>
#include <math.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
//#include <util/keyval/keyvalass.h>

using namespace sc;

class TensorRank4 {

public:
	TensorRank4(int dim0, int dim1, int dim2, int dim3) {
		dims_[0] = dim0;
		dims_[1] = dim1;
		dims_[2] = dim2;
		dims_[3] = dim3;
		data_.resize(dims_[0] * dims_[1] * dims_[2] * dims_[3]);
	}

	double& operator ()(int i, int j, int k, int l) {
		return data_(index(i, j, k, l));
	}

	const double& operator ()(int i, int j, int k, int l) const {
		return data_(index(i, j, k, l));
	}

private:

	int index(int i, int j, int k, int l) const {
		return i * dims_[2] * dims_[1] * dims_[0] + j * dims_[1] * dims_[0]
				+ k * dims_[0] + l;
	}
	size_t dims_[4];
	Eigen::VectorXd data_;
};

class TensorRank6 {

public:
	TensorRank6(int dim0, int dim1, int dim2, int dim3, int dim4, int dim5) {
		dims_[0] = dim0;
		dims_[1] = dim1;
		dims_[2] = dim2;
		dims_[3] = dim3;
		dims_[4] = dim4;
		dims_[5] = dim5;
		data_.resize(
				dims_[0] * dims_[1] * dims_[2] * dims_[3] * dims_[4]
						* dims_[5]);
	}

	double& operator ()(int i, int j, int k, int l, int m, int n) {
		return data_(index(i, j, k, l, m, n));
	}

	const double& operator ()(int i, int j, int k, int l, int m, int n) const {
		return data_(index(i, j, k, l, m, n));
	}

private:

	int index(int i, int j, int k, int l, int m, int n) const {
		return i * dims_[4] * dims_[3] * dims_[2] * dims_[1] * dims_[0]
				+ j * dims_[3] * dims_[2] * dims_[1] * dims_[0]
				+ k * dims_[2] * dims_[1] * dims_[0] + l * dims_[1] * dims_[0]
				+ m * dims_[0] + n;
	}
	size_t dims_[6];
	Eigen::VectorXd data_;
};

void get_overlap_ints(Ref<OneBodyInt> s_inteval, Eigen::MatrixXd &S_mat) {

	const double* buffer = s_inteval->buffer();

	const int nshell = s_inteval->basis1()->nshell();
	std::cout << "overlap integrals:" << std::endl;

	for (int s1 = 0; s1 < nshell; s1++) {
		const int bf1_offset = s_inteval->basis1()->shell_to_function(s1);
		const int nbf1 = s_inteval->basis1()->shell(s1).nfunction();

		for (int s2 = 0; s2 < nshell; s2++) {
			const int bf2_offset = s_inteval->basis1()->shell_to_function(s2);
			const int nbf2 = s_inteval->basis1()->shell(s2).nfunction();

			s_inteval->compute_shell(s1, s2);

			int bf12 = 0;
			for (int bf1 = 0; bf1 < nbf1; ++bf1) {
				for (int bf2 = 0; bf2 < nbf2; ++bf2, ++bf12) {

					S_mat(bf1 + bf1_offset, bf2 + bf2_offset) = buffer[bf12];

				}
			}
		}
	}
}

void get_core_hamiltonian_ints(Ref<OneBodyInt> h_inteval,
		Eigen::MatrixXd &Hcore_mat) {

	const double* buffer = h_inteval->buffer();
	const int nshell = h_inteval->basis1()->nshell();

	std::cout << "hcore integrals:" << std::endl;
	for (int s1 = 0; s1 < nshell; s1++) {
		const int bf1_offset = h_inteval->basis1()->shell_to_function(s1);
		const int nbf1 = h_inteval->basis1()->shell(s1).nfunction();

		for (int s2 = 0; s2 < nshell; s2++) {
			const int bf2_offset = h_inteval->basis1()->shell_to_function(s2);
			const int nbf2 = h_inteval->basis1()->shell(s2).nfunction();

			h_inteval->compute_shell(s1, s2);

			int bf12 = 0;
			for (int bf1 = 0; bf1 < nbf1; ++bf1) {
				for (int bf2 = 0; bf2 < nbf2; ++bf2, ++bf12) {
					//      std::cout << bf1+bf1_offset << " " << bf2+bf2_offset << " "
					//        << std::setprecision(15) << buffer[bf12] << std::endl;
					Hcore_mat(bf1 + bf1_offset, bf2 + bf2_offset) =
							buffer[bf12];
				}
			}
		}
	}
}

void get_two_electron_ints(Ref<TwoBodyInt> twoecoulomb_inteval,
		Eigen::MatrixXd &Heffective_mat) {

	const double* buffer = twoecoulomb_inteval->buffer();
	const int nshell = twoecoulomb_inteval->basis1()->nshell();
	const int nbasis = twoecoulomb_inteval->basis1()->nbasis();

	std::cout << "two-e Coulomb integrals:" << std::endl;
	for (int s1 = 0; s1 < nshell; s1++) {
		const int bf1_offset = twoecoulomb_inteval->basis1()->shell_to_function(
				s1);
		const int nbf1 = twoecoulomb_inteval->basis1()->shell(s1).nfunction();

		for (int s2 = 0; s2 < nshell; s2++) {
			const int bf2_offset =
					twoecoulomb_inteval->basis1()->shell_to_function(s2);
			const int nbf2 =
					twoecoulomb_inteval->basis1()->shell(s2).nfunction();

			for (int s3 = 0; s3 < nshell; s3++) {
				const int bf3_offset =
						twoecoulomb_inteval->basis1()->shell_to_function(s3);
				const int nbf3 =
						twoecoulomb_inteval->basis1()->shell(s3).nfunction();

				for (int s4 = 0; s4 < nshell; s4++) {
					const int bf4_offset =
							twoecoulomb_inteval->basis1()->shell_to_function(
									s4);
					const int nbf4 =
							twoecoulomb_inteval->basis1()->shell(s4).nfunction();

					twoecoulomb_inteval->compute_shell(s1, s2, s3, s4);

					int bf1234 = 0;
					for (int bf1 = 0; bf1 < nbf1; ++bf1) {
						for (int bf2 = 0; bf2 < nbf2; ++bf2) {
							for (int bf3 = 0; bf3 < nbf3; ++bf3) {
								for (int bf4 = 0; bf4 < nbf4; ++bf4, ++bf1234) {

									Heffective_mat(
											(bf1 + bf1_offset) * nbasis + bf2
													+ bf2_offset,
											(bf3 + bf3_offset) * nbasis + bf4
													+ bf4_offset) =
											buffer[bf1234];
								}
							}
						}
					}
				}
			}
		}
	}
}

Eigen::MatrixXd symmetric_orthogonalization(const Eigen::MatrixXd &S) {
	//Symmetric orthogonalization
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolverS(S);
	Eigen::MatrixXd X = eigensolverS.operatorInverseSqrt();

	return X;
}

Eigen::MatrixXd cannonical_orthogonalization(const Eigen::MatrixXd &S,
		const int nbasis) {
	//Canonical orthogonalization
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolverS(S);

	Eigen::MatrixXd U = eigensolverS.eigenvectors();
	Eigen::MatrixXd Ut = U.transpose();
	Eigen::VectorXd s = eigensolverS.eigenvalues();

	Eigen::ArrayXd sa = s.array();
	Eigen::ArrayXd ss = sa.sqrt();

	//sqrts is s^(1/2)
	Eigen::MatrixXd sqrts(nbasis, nbasis);

	for (size_t i = 0; i < nbasis; i++) {
		for (size_t j = 0; j < nbasis; j++) {
			if (i == j) {
				sqrts(i, j) = ss(i);
			} else {
				sqrts(i, j) = 0;
			}
		}
	}

	Eigen::MatrixXd X = U * sqrts.inverse();

	return X;
}

Eigen::MatrixXd core_hamiltonian_transformation(const Eigen::MatrixXd &H,
		const Eigen::MatrixXd &X, const Eigen::MatrixXd &Xt) {
	Eigen::MatrixXd H_t = Xt * H * X;
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolverH(H);
	Eigen::MatrixXd C_t = eigensolverH.eigenvectors();
	Eigen::MatrixXd C = X * C_t;

	return C;
}

Eigen::MatrixXd hamiltonian_transformation(const Eigen::MatrixXd &H,
		const Eigen::MatrixXd &X, const Eigen::MatrixXd &Xt) {
	Eigen::MatrixXd H_t = Xt * H * X;
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolverH(H_t);
	Eigen::MatrixXd C_t = eigensolverH.eigenvectors();
	Eigen::MatrixXd C = X * C_t;

	return C;
}

Eigen::MatrixXd c_occupied(const Eigen::MatrixXd &C, const int nocc) {

	Eigen::MatrixXd c(C.cols(), nocc);
	for (size_t i = 0; i < C.cols(); i++) {
		for (size_t j = 0; j < nocc; j++) {
			c(i, j) = C(i, j);
		}
	}

	return c;
}

Eigen::MatrixXd fock_build(const Eigen::MatrixXd &H_core,
		const Eigen::MatrixXd &H_effective, const Eigen::MatrixXd &P,
		const int nbasis) {
	Eigen::MatrixXd F(nbasis, nbasis);

	Eigen::MatrixXd J_ao(nbasis, nbasis), K_ao(nbasis, nbasis);
	Eigen::MatrixXd J(nbasis, nbasis), K(nbasis, nbasis);

	for (size_t i = 0; i < nbasis; i++) {
		for (size_t j = 0; j < nbasis; j++) {

			for (size_t k = 0; k < nbasis; k++) {
				for (size_t l = 0; l < nbasis; l++) {
					J_ao(k, l) = H_effective(i * nbasis + j, l * nbasis + k);
					K_ao(k, l) = H_effective(i * nbasis + k, l * nbasis + j);
				}
			}

			J(i, j) = (P * J_ao).trace();
			K(i, j) = (P * K_ao).trace();

			F(i, j) = H_core(i, j) + J(i, j) - 0.5 * K(i, j);

		}
	}

	return F;
}

double electronic_energy(const Eigen::MatrixXd &H_core,
		const Eigen::MatrixXd &F, const Eigen::MatrixXd &P_new,
		const int nbasis) {
	double E = 0.0;
	for (size_t i = 0; i < nbasis; i++) {
		for (size_t j = 0; j < nbasis; j++) {
			E += P_new(i, j) * (H_core(i, j) + F(i, j));
		}
	}

	return E;
}

double rms_density_norm(const Eigen::MatrixXd &P, const Eigen::MatrixXd &P_new,
		const int nbasis) {
	double diff = 0;
	for (size_t i = 0; i < nbasis; i++) {
		for (size_t j = 0; j < nbasis; j++) {
			diff += (pow(P_new(i, j) - P(i, j), 2.0));
		}
	}
	return sqrt(diff);
}

Eigen::VectorXd orbital_energies(const Eigen::MatrixXd &F,
		const Eigen::MatrixXd &X, const Eigen::MatrixXd &Xt) {
	Eigen::MatrixXd F_t = Xt * F * X;
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolverF(F_t);
	Eigen::MatrixXd e_orbital = eigensolverF.eigenvalues();

	return e_orbital;
}

Eigen::MatrixXd HartreeFock(const Eigen::MatrixXd &S,
		const Eigen::MatrixXd &H_core, const Eigen::MatrixXd &H_effective,
		const int nbasis, const int nocc, const double nuc_repulsion,
		Eigen::MatrixXd &F, Eigen::MatrixXd &X) {

	X = symmetric_orthogonalization(S);
	//Eigen::MatrixXd X = cannonical_orthogonalization(S, nbasis);
	Eigen::MatrixXd Xt = X.transpose();

	//Obtaining C coefficients from H_core_t
	Eigen::MatrixXd C = core_hamiltonian_transformation(H_core, X, Xt);

	//Obtaining c coefficients of occupied orbitals
	Eigen::MatrixXd c = c_occupied(C, nocc);

	//Generating density matrix
	Eigen::MatrixXd P = 2 * c * c.transpose();

	//std::cout << P << std::endl;

	//SCF procedure
	int count = 0;
	double difference = 1;
	const double tollerance = 1e-6;
	Eigen::MatrixXd C_new(nbasis, nbasis);

	while (difference > tollerance) {

		count++;

		//Creating fock matrix
		F = fock_build(H_core, H_effective, P, nbasis);

		//std::cout << F << std::endl;

		C_new = hamiltonian_transformation(F, X, Xt);

		Eigen::MatrixXd c_new = c_occupied(C_new, nocc);

		Eigen::MatrixXd P_new = 2.0 * c_new * c_new.transpose();

		double E_electronic = electronic_energy(H_core, F, P_new, nbasis);

		difference = rms_density_norm(P, P_new, nbasis);

		double E_tot = 0.5 * E_electronic + nuc_repulsion;

		std::cout << "iter " << count << "\t" << std::setprecision(10)
				<< " Energy = " << E_tot << "\t" << "error = " << difference
				<< std::endl;

		P = P_new;
		//P = 0.8 * P + 0.2 * P_new;
	}

	//std::cout << e_orbital << std::endl;
	return C_new;
}

TensorRank4 IntegralTransformation(const Eigen::MatrixXd &H_effective,
		const Eigen::MatrixXd &C, const int nbasis) {

	TensorRank4 orb(nbasis, nbasis, nbasis, nbasis);

	Eigen::MatrixXd g = H_effective;
	g.resize(nbasis, nbasis * nbasis * nbasis); //g(p,qrs)

	Eigen::MatrixXd t1 = g.transpose() * C; //t1(qrs,i)
	t1.resize(nbasis, nbasis * nbasis * nbasis); //t1(q,rsi)

	Eigen::MatrixXd t2 = t1.transpose() * C; //t2(rsi,j)
	t2.resize(nbasis, nbasis * nbasis * nbasis); //t2(r,sij)

	Eigen::MatrixXd t3 = t2.transpose() * C; //t3(sij,k)
	t3.resize(nbasis, nbasis * nbasis * nbasis); //t2(s,ijk)

	Eigen::MatrixXd t4 = t3.transpose() * C; //t3(ijk,l)
	t4.resize(nbasis, nbasis * nbasis * nbasis); //t2(i,jkl)

	t4.resize(nbasis * nbasis, nbasis * nbasis);

	for (size_t i = 0; i < nbasis; i++) {
		for (size_t j = 0; j < nbasis; j++) {
			for (size_t k = 0; k < nbasis; k++) {
				for (size_t l = 0; l < nbasis; l++) {
					orb(i, j, k, l) = t4(i * nbasis + j, k * nbasis + l);
				}
			}
		}
	}

	return orb;
}

TensorRank4 MP2_second_quantization(const TensorRank4 &orb, const int nocc,
		const int nbasis, const Eigen::MatrixXd &F, const Eigen::MatrixXd &X,
		TensorRank4 &dij, TensorRank4 &aij) {

	TensorRank4 Tij(nbasis, nbasis, nbasis, nbasis);
	TensorRank4 Tji(nbasis, nbasis, nbasis, nbasis);
	TensorRank4 Tij_tilda(nbasis, nbasis, nbasis, nbasis);

	Eigen::MatrixXd Xt = X.transpose();
	Eigen::VectorXd e_orbital = orbital_energies(F, X, Xt);
	Eigen::MatrixXd Ft = Xt * F * X;

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolverFt(Ft);
	Eigen::MatrixXd U = eigensolverFt.eigenvectors();
	Eigen::MatrixXd FT = U.transpose() * Ft * U;

	//std::cout << FT << std::endl;

	const int nvirt = nbasis - nocc;
	//transformation of Fock matrix (virtual space)
	Eigen::MatrixXd F_virt(nvirt, nvirt);
	F_virt = Eigen::MatrixXd::Zero(nvirt, nvirt);
	Eigen::VectorXd e_virtual(nvirt);

	Eigen::MatrixXd E_MP2_pair(nocc, nocc);
	E_MP2_pair = Eigen::MatrixXd::Zero(nocc, nocc);

	double E_MP2 = 0;
	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {

			double summ = 0;
			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					Tij(i, a, j, b) = -orb(i, a, j, b)
							/ (e_orbital(a) + e_orbital(b) - e_orbital(i)
									- e_orbital(j));
					Tji(i, a, j, b) = -orb(i, b, j, a)
							/ (e_orbital(a) + e_orbital(b) - e_orbital(i)
									- e_orbital(j));
					Tij_tilda(i, a, j, b) = 2.0 * Tij(i, a, j, b)
							- 1.0 * Tji(i, a, j, b);

					summ += orb(i, a, j, b) * Tij_tilda(i, a, j, b);
				}
			}
			E_MP2_pair(i, j) = summ;
			//std::cout << "Energy of pair " << std::endl;
			//std::cout << E_MP2_pair(i, j) << std::endl;

			E_MP2 += E_MP2_pair(i, j);
		}
	}

	//LPNO
	//based on paper F. Neese et al., J. Chem. Phys. 130, 114108 (2009)
	TensorRank4 Dij(nbasis, nbasis, nbasis, nbasis);

	TensorRank4 Tij_tilda_PNO(nbasis, nbasis, nbasis, nbasis);
	TensorRank4 Tij_PNO(nbasis, nbasis, nbasis, nbasis);
	TensorRank4 Tji_PNO(nbasis, nbasis, nbasis, nbasis);
	TensorRank4 K_PNO(nbasis, nbasis, nbasis, nbasis);

	TensorRank4 K_tilda(nbasis, nbasis, nbasis, nbasis);

	TensorRank4 Tij_tilda_PNO_T(nbasis, nbasis, nbasis, nbasis);
	TensorRank4 Tij_PNO_T(nbasis, nbasis, nbasis, nbasis);
	TensorRank4 Tji_PNO_T(nbasis, nbasis, nbasis, nbasis);
	TensorRank4 K_PNO_T(nbasis, nbasis, nbasis, nbasis);
	TensorRank4 K_tilda_T(nbasis, nbasis, nbasis, nbasis);

	Eigen::MatrixXd E_MP2_pair_PNO(nocc, nocc);
	E_MP2_pair_PNO = Eigen::MatrixXd::Zero(nocc, nocc);

	Eigen::MatrixXd E_MP2_pair_PNO_T(nocc, nocc);
	E_MP2_pair_PNO_T = Eigen::MatrixXd::Zero(nocc, nocc);

	//defined like in orca mdci_pno.cpp under TPNOPairData_RHF::MakePairSpecificPNOs_RHF function
	Eigen::MatrixXd E_MP2_pair_Full(nocc, nocc);
	E_MP2_pair_Full = Eigen::MatrixXd::Zero(nocc, nocc);

	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {

			Eigen::MatrixXd K_mat(nvirt, nvirt), Kij_PNO(nvirt, nvirt), Jii(
					nvirt, nvirt), Jjj(nvirt, nvirt), Kii(nvirt, nvirt), Kjj(
					nvirt, nvirt);
			K_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Kij_PNO = Eigen::MatrixXd::Zero(nvirt, nvirt);

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					Jii(a - nocc, b - nocc) = orb(i, i, a, b);
					Jjj(a - nocc, b - nocc) = orb(j, j, a, b);
					Kii(a - nocc, b - nocc) = orb(i, a, i, b);
					Kjj(a - nocc, b - nocc) = orb(j, a, j, b);
					K_mat(a - nocc, b - nocc) = orb(i, a, j, b);
				}
			}

			//Meyer transformation
			Eigen::MatrixXd Gij(nvirt, nvirt);
			Gij = Eigen::MatrixXd::Zero(nvirt, nvirt);

			if (i == j) {
				for (size_t a = nocc; a < nbasis; a++) {
					for (size_t b = nocc; b < nbasis; b++) {
						Gij(a - nocc, b - nocc) = FT(a, b)
								+ Kii(a - nocc, b - nocc)
								- Jii(a - nocc, b - nocc);
					}
				}
			} else {
				for (size_t a = nocc; a < nbasis; a++) {
					for (size_t b = nocc; b < nbasis; b++) {
						Gij(a - nocc, b - nocc) = FT(a, b)
								+ Kii(a - nocc, b - nocc)
								+ Kjj(a - nocc, b - nocc)
								- 0.5 * Jii(a - nocc, b - nocc)
								- 0.5 * Jjj(a - nocc, b - nocc);
					}
				}
			}

			//std::cout << "G matrix \n"<<Gij << std::endl;

			//Diagonalization of G matrix

			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolverGij(Gij);
			Eigen::MatrixXd GV = eigensolverGij.eigenvectors();
			Eigen::VectorXd GE = eigensolverGij.eigenvalues();

			//transforming K_mat into Kij_new using GV
			Eigen::MatrixXd Kij_new = GV.transpose() * K_mat * GV;

			//std::cout << "Kij_new matrix \n"<< K_mat << std::endl;

			Eigen::MatrixXd Tij_new(nvirt, nvirt);
			Tij_new = Eigen::MatrixXd::Zero(nvirt, nvirt);

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					Tij_new(a - nocc, b - nocc) = -Kij_new(a - nocc, b - nocc)
							/ (GE(a - nocc) + GE(b - nocc) - e_orbital(i)
									- e_orbital(j));
				}
			}

			//std::cout << "Tij_new matrix \n"<< Tij_new << std::endl;

			//transformation of Tij_new with GV
			Eigen::MatrixXd Tij_t = GV * Tij_new * GV.transpose();
			//std::cout << "Tij_t matrix \n"<< Tij_t << std::endl;

			//transormation of exchange integrals in PNO basis: Kij^ab (ij are in subscript & ab in upperscript)

			Eigen::MatrixXd Tij_mat(nvirt, nvirt), Tij_tilda_mat(nvirt, nvirt),
					Dij_mat(nvirt, nvirt);
			Tij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Tij_tilda_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Dij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					Tij_mat(a - nocc, b - nocc) = Tij_t(a - nocc, b - nocc);
					if (i == j) {
						Tij_tilda_mat(a - nocc, b - nocc) = 2.0
								* Tij_t(a - nocc, b - nocc)
								- 1.0 * Tij_t(b - nocc, a - nocc);
					} else {
						Tij_tilda_mat(a - nocc, b - nocc) = 4.0
								* Tij_t(a - nocc, b - nocc)
								- 2.0 * Tij_t(b - nocc, a - nocc);
					}
				}
			}

			//E_MP2_pair_Full from orca mdci_pno

			double summa = 0.0;
			if (i == j) {
				for (size_t a = nocc; a < nbasis; a++) {
					for (size_t b = nocc; b < nbasis; b++) {
						summa += Tij_t(a - nocc, b - nocc)
								* K_mat(a - nocc, b - nocc);
					}
				}
			} else {
				for (size_t a = nocc; a < nbasis; a++) {
					for (size_t b = nocc; b < nbasis; b++) {
						summa += 2.0 * Tij_t(a - nocc, b - nocc)
								* (2.0 * K_mat(a - nocc, b - nocc)
										- K_mat(b - nocc, a - nocc));
					}
				}
			}
			E_MP2_pair_Full(i, j) = summa;

			//std::cout << "Tij_tilda_mat amplitudes \n"<< Tij_tilda_mat << "\n"<< std::cout;

			//computation of <Tij_tilda*Tij> = t
			//from paper

			double t = 0.0;
			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					t += Tij_tilda_mat(b - nocc, a - nocc)
							* Tij_mat(a - nocc, b - nocc);
				}
			}

			double Nij = 1.0 + t;

			//std::cout << Nij << std::endl;

			if (i == j) {
				Dij_mat = (1.0 + 1.0) / Nij
						* (Tij_tilda_mat.transpose() * Tij_mat
								+ Tij_tilda_mat * Tij_mat.transpose());
			} else {
				Dij_mat = 1.0 / Nij
						* (Tij_tilda_mat.transpose() * Tij_mat
								+ Tij_tilda_mat * Tij_mat.transpose());
			};

			Eigen::MatrixXd dij_mat(nvirt, nvirt);
			dij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::VectorXd aij_vect(nvirt);
			aij_vect = Eigen::VectorXd::Zero(nvirt);

			Eigen::MatrixXd aij_mat(nvirt, nvirt);
			aij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolverDij_mat(
					Dij_mat);
			dij_mat = eigensolverDij_mat.eigenvectors();
			aij_vect = eigensolverDij_mat.eigenvalues();

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					if (a == b) {
						aij_mat(a - nocc, b - nocc) = aij_vect(a - nocc);
					} else {
						aij_mat(a - nocc, b - nocc) = 0;
					}
				}
			}

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					Dij(i, a, j, b) = Dij_mat(a - nocc, b - nocc);
					dij(i, a, j, b) = dij_mat(a - nocc, b - nocc);
					aij(i, a, j, b) = aij_mat(a - nocc, b - nocc);
				}
			}

			std::cout << "PNO occupation number for pair i,j = " << i << ","
					<< j << std::endl;
			std::cout << aij_vect << "\n" << std::endl;

			//
			//transformation to PNO basis
			//
			//transformation of Kij_new exchange integrals into PNO basis

			/*
			 for (size_t a = nocc; a < nbasis; a++) {
			 for (size_t b = nocc; b < nbasis; b++) {
			 double num = 0;
			 for (size_t c = 0; c < nvirt; c ++) {
			 for (size_t d = 0; d < nvirt; d ++) {
			 num += dij_mat(c, a - nocc)*K_mat(c, d)*dij_mat(d, b - nocc);
			 }
			 }
			 Kij_PNO(a - nocc, b - nocc) = num;
			 }
			 }
			 */
			//transformation of K_matrix into PNO basis
			Kij_PNO = dij_mat.transpose() * K_mat * dij_mat;

			//Construction of fock operator of virtual canonical orbitals

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					F_virt(a - nocc, b - nocc) = FT(a, b);
				}
			}

			//transformation of F_virt into PNO space
			Eigen::MatrixXd F_virt_PNO = dij_mat.transpose() * F_virt * dij_mat;

			//std::cout << F_virt_PNO << std::endl;

			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolverF_virt_PNO(
					F_virt_PNO);
			Eigen::MatrixXd L = eigensolverF_virt_PNO.eigenvectors();
			e_virtual = eigensolverF_virt_PNO.eigenvalues();

			Eigen::MatrixXd K_mat_tilda(nvirt, nvirt);
			K_mat_tilda = Eigen::MatrixXd::Zero(nvirt, nvirt);
			K_mat_tilda = L.transpose() * Kij_PNO * L;

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					K_tilda(i, a, j, b) = K_mat_tilda(a - nocc, b - nocc);
				}
			}

			//E pair ij
			double summ = 0;
			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					Tij_PNO(i, a, j, b) = -K_tilda(i, a, j, b)
							/ (e_virtual(a - nocc) + e_virtual(b - nocc)
									- e_orbital(i) - e_orbital(j));
					Tji_PNO(i, a, j, b) = -K_tilda(i, b, j, a)
							/ (e_virtual(a - nocc) + e_virtual(b - nocc)
									- e_orbital(i) - e_orbital(j));
					Tij_tilda_PNO(i, a, j, b) = 2.0 * Tij_PNO(i, a, j, b)
							- 1.0 * Tji_PNO(i, a, j, b);

					summ += K_tilda(i, a, j, b) * Tij_tilda_PNO(i, a, j, b);
				}
			}

			E_MP2_pair_PNO(i, j) = summ;

			std::cout << "Energy of pair Full" << std::endl;
			std::cout << E_MP2_pair_Full(i, j) << "\n" << std::endl;

			//MP2-PNO energy after PNO truncation for ij pair
			//Truncation criteria is TCutPNO < 1e-7 & TCutPairs < 1e-4
			double TCutPNO = 1e-15;
			double TCutPairs = 0.0; //Hartree
			int dimPNO = 0;

			double Pair_En;
			if (E_MP2_pair_Full(i, j) < 0) {
				Pair_En = -1.0 * E_MP2_pair_Full(i, j);
			} else {
				Pair_En = 1.0 * E_MP2_pair_Full(i, j);
			}

			if (Pair_En > TCutPairs) {

				//counting a number of PNO's for a given pair
				if (TCutPNO == 0) {
					dimPNO = nvirt;
				} else {
					for (size_t p = 0; p < nvirt; p++) {
						if (aij_vect(p) > TCutPNO) {
							dimPNO += 1;
						}
					}
				}

				//std::cout << "number of PNO's per pair ij" << std::endl;
				//std::cout << dimPNO << std::endl;

				//storing PNO coefficients dij after truncation

				//aij_vec_T & dij_mat_T are PNO occupation number and coefficients after truncation for given ij pair
				Eigen::MatrixXd dij_mat_T1(nvirt, dimPNO);
				dij_mat_T1 = Eigen::MatrixXd::Zero(nvirt, dimPNO);

				Eigen::MatrixXd dij_mat_T(nvirt, dimPNO + 1);
				dij_mat_T = Eigen::MatrixXd::Zero(nvirt, dimPNO + 1);

				Eigen::VectorXd aij_vect_T(dimPNO);
				aij_vect_T = Eigen::VectorXd::Zero(dimPNO);

				for (size_t p = 0; p < dimPNO; p++) {
					aij_vect_T(p) = aij_vect(nvirt - dimPNO + p);

					for (size_t a = 0; a < nvirt; a++) {
						dij_mat_T1(a, p) = dij_mat(a, nvirt - dimPNO + p);
					}
				}

				//std::cout << "size of dij truncated" << std::endl;
				//std::cout << dij_mat_T.cols() << std::endl;

				if (dij_mat_T1.cols() != 0) {
					dij_mat_T = dij_mat_T1;
					//std::cout << "dij truncated" << std::endl;
					//std::cout << dij_mat_T << std::endl;
					//transformation of Kij & Tij with truncated dij and energy calculation of MP2-PNO-truncated

					Eigen::MatrixXd Kij_PNO_T(dimPNO, dimPNO);
					Kij_PNO_T = Eigen::MatrixXd::Zero(dimPNO, dimPNO);
					//transformation of K_matrix into truncated PNO basis
					Kij_PNO_T = dij_mat_T.transpose() * K_mat * dij_mat_T;

					//transformation of F_virt into truncated PNO space
					Eigen::MatrixXd F_virt_PNO_T = dij_mat_T.transpose()
							* F_virt * dij_mat_T;

					Eigen::VectorXd e_virtual_T(dimPNO);

					Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolverF_virt_PNO_T(
							F_virt_PNO_T);

					Eigen::MatrixXd L_T =
							eigensolverF_virt_PNO_T.eigenvectors();
					e_virtual_T = eigensolverF_virt_PNO_T.eigenvalues();

					Eigen::MatrixXd K_mat_tilda_T(dimPNO, dimPNO);
					K_mat_tilda_T = Eigen::MatrixXd::Zero(dimPNO, dimPNO);
					K_mat_tilda_T = L_T.transpose() * Kij_PNO_T * L_T;

					for (size_t a = nocc; a < nocc + dimPNO; a++) {
						for (size_t b = nocc; b < nocc + dimPNO; b++) {
							K_tilda_T(i, a, j, b) = K_mat_tilda_T(a - nocc,
									b - nocc);
						}
					}

					//E pair ij
					double summ_T = 0;
					for (size_t a = nocc; a < nocc + dimPNO; a++) {
						for (size_t b = nocc; b < nocc + dimPNO; b++) {
							Tij_PNO_T(i, a, j, b) = -K_tilda_T(i, a, j, b)
									/ (e_virtual_T(a - nocc)
											+ e_virtual_T(b - nocc)
											- e_orbital(i) - e_orbital(j));
							Tji_PNO_T(i, a, j, b) = -K_tilda_T(i, b, j, a)
									/ (e_virtual_T(a - nocc)
											+ e_virtual_T(b - nocc)
											- e_orbital(i) - e_orbital(j));
							Tij_tilda_PNO_T(i, a, j, b) = 2.0
									* Tij_PNO_T(i, a, j, b)
									- 1.0 * Tji_PNO_T(i, a, j, b);

							summ_T += K_tilda_T(i, a, j, b)
									* Tij_tilda_PNO_T(i, a, j, b);
						}
					}

					E_MP2_pair_PNO_T(i, j) = summ_T;
				}

			}

		}
	}

	double E_MP2_tilda = 0.0;
	double E_MP2_tilda_T = 0.0;
	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {

			E_MP2_tilda += E_MP2_pair_PNO(i, j);
			E_MP2_tilda_T += E_MP2_pair_PNO_T(i, j);

		}
	}

	double E_MP2_new = 0.0;
	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j <= i; j++) {
			E_MP2_new += E_MP2_pair_Full(i, j);
		}
	}

	std::cout << "MP2 energy" << std::endl;
	std::cout << E_MP2 << std::endl;
	std::cout << "PNO MP2 energy without truncation" << std::endl;
	std::cout << E_MP2_tilda << std::endl;
	std::cout << "PNO MP2 energy with truncation" << std::endl;
	std::cout << E_MP2_tilda_T << std::endl;

	return Tij;

}

TensorRank4 get_residual(const TensorRank4 &orb, const TensorRank4 Tij_n,
		const Eigen::MatrixXd &Ft, const int nocc, const int nbasis) {

	TensorRank4 Res(nbasis, nbasis, nbasis, nbasis);

	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {
			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {

					double summ_c = 0;
					for (size_t c = nocc; c < nbasis; c++) {
						summ_c += Ft(a, c) * Tij_n(i, c, j, b)
								+ Tij_n(i, a, j, c) * Ft(c, b);
					}
					double summ_k = 0;
					for (size_t k = 0; k < nocc; k++) {
						summ_k += Ft(i, k) * Tij_n(k, a, j, b)
								+ Tij_n(i, a, k, b) * Ft(k, j);
					}
					Res(i, a, j, b) = orb(i, a, j, b) + summ_c - summ_k;
					//Res(i, a, j, b) = orb(i, a, j, b) - summ_k;

				}
			}
		}
	}

	return Res;
}

TensorRank4 get_increment_of_amplitude(const TensorRank4 &Res,
		const Eigen::MatrixXd &Ft, const int nocc, const int nbasis) {
	TensorRank4 dT(nbasis, nbasis, nbasis, nbasis);

	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {
			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {

					dT(i, a, j, b) = -Res(i, a, j, b)
							/ (Ft(a, a) + Ft(b, b) - Ft(i, i) - Ft(j, j));
				}
			}
		}
	}

	return dT;
}

TensorRank4 get_new_aplitudes(TensorRank4 &Tij_n, const TensorRank4 &dT,
		const int nocc, const int nbasis) {

	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {
			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {

					Tij_n(i, a, j, b) = Tij_n(i, a, j, b) + dT(i, a, j, b);
				}
			}
		}
	}

	return Tij_n;
}

double max_abs_Res(TensorRank4 &Res, const int nocc, const int nbasis) {

	double max = 0.0;

	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {
			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					max = std::max(max, std::abs(Res(i, a, j, b)));
				}
			}
		}
	}
	return max;
}

void MP2_linear_system_solver(const TensorRank4 &orb, const Eigen::MatrixXd &F,
		const Eigen::MatrixXd &C, TensorRank4 Tij, const int nocc,
		const int nbasis) {

	Eigen::MatrixXd Ft = C.transpose() * F * C;
	//std::cout << Ft << std::endl;

	TensorRank4 Tij_n(nbasis, nbasis, nbasis, nbasis);
	TensorRank4 Res(nbasis, nbasis, nbasis, nbasis);

	double nvirt = nbasis - nocc;

	//initial guess of amplitudes
	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {
			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					Tij_n(i, a, j, b) = Tij(i, a, j, b);
					//std::cout << Tij(i, a, j, b) << std::endl;
				}
			}
		}
	}

	int count = 0;
	while (true) {

		TensorRank4 Res = get_residual(orb, Tij_n, Ft, nocc, nbasis);

		//increment of T
		TensorRank4 dT = get_increment_of_amplitude(Res, Ft, nocc, nbasis);

		//updating new amplitudes as Tij_n = Tij_n + dT
		Tij_n = get_new_aplitudes(Tij_n, dT, nocc, nbasis);

		double max = max_abs_Res(Res, nocc, nbasis);

		count++;
		if (max < 1e-10)
			break;
	}

	double E_MP2 = 0;
	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {

			double summ = 0;
			double E_MP2_pair = 0.0;
			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {

					summ += orb(i, a, j, b)
							* (2.0 * Tij_n(i, a, j, b) - Tij_n(i, b, j, a));
				}
			}
			E_MP2_pair = summ;

			E_MP2 += E_MP2_pair;
		}
	}

	std::cout << "MP2 energy linear systems 2" << std::endl;
	std::cout << E_MP2 << std::endl;
	std::cout << count << std::endl;
}

TensorRank4 transformation_of_doubles_amplitudes_from_MO_into_PNO(
		TensorRank4 Tij, const TensorRank4 dij, const int nocc,
		const int nbasis) {

	TensorRank4 Tij_PNO(nbasis, nbasis, nbasis, nbasis);
	const int nvirt = nbasis - nocc;

	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {

			Eigen::MatrixXd dij_mat(nvirt, nvirt);
			dij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::MatrixXd Tij_mat(nvirt, nvirt);
			Tij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					dij_mat(a - nocc, b - nocc) = dij(i, a, j, b);
					Tij_mat(a - nocc, b - nocc) = Tij(i, a, j, b);
				}
			}

			Eigen::MatrixXd Tij_mat_PNO = dij_mat * Tij_mat
					* dij_mat.transpose();

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					Tij_PNO(i, a, j, b) = Tij_mat_PNO(a - nocc, b - nocc);
				}
			}

		}
	}

	return Tij_PNO;
}

TensorRank4 get_residual_PNO(const TensorRank4 &orb, const TensorRank4 Tij_PNO,
		TensorRank4 &dij, const Eigen::MatrixXd &Ft, const Eigen::MatrixXd &C,
		const int nocc, const int nbasis) {

	TensorRank4 Res_PNO(nbasis, nbasis, nbasis, nbasis);

	const int nvirt = nbasis - nocc;

	Eigen::MatrixXd F_virt(nvirt, nvirt);

	for (size_t a = nocc; a < nbasis; a++) {
		for (size_t b = nocc; b < nbasis; b++) {
			F_virt(a - nocc, b - nocc) = Ft(a, b);
		}
	}

	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {

			Eigen::MatrixXd R_mat(nvirt, nvirt);
			R_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd dij_mat(nvirt, nvirt);
			dij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::MatrixXd dijt_mat(nvirt, nvirt);
			dijt_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::MatrixXd K_mat(nvirt, nvirt);
			K_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::MatrixXd Tij_mat_PNO(nvirt, nvirt);
			Tij_mat_PNO = Eigen::MatrixXd::Zero(nvirt, nvirt);

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					dij_mat(a - nocc, b - nocc) = dij(i, a, j, b);
					dijt_mat(a - nocc, b - nocc) = dij(i, b, j, a);
					K_mat(a - nocc, b - nocc) = orb(i, a, j, b);
					Tij_mat_PNO(a - nocc, b - nocc) = Tij_PNO(i, a, j, b);
				}
			}

			Eigen::MatrixXd term_1 = dijt_mat * K_mat * dij_mat;

			Eigen::MatrixXd term_3 = dijt_mat * F_virt.transpose() * dij_mat
					* Tij_mat_PNO + Tij_mat_PNO * dijt_mat * F_virt * dij_mat;

			Eigen::MatrixXd term_4(nvirt, nvirt);
			term_4 = Eigen::MatrixXd::Zero(nvirt, nvirt);

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {

					double summ1 = 0.0;
					for (size_t k = 0; k < nocc; k++) {

						Eigen::MatrixXd dik_mat(nvirt, nvirt);
						dik_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd dkj_mat(nvirt, nvirt);
						dkj_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd Tik_mat_PNO(nvirt, nvirt);
						Tik_mat_PNO = Eigen::MatrixXd::Zero(nvirt, nvirt);
						Eigen::MatrixXd Tkj_mat_PNO(nvirt, nvirt);
						Tkj_mat_PNO = Eigen::MatrixXd::Zero(nvirt, nvirt);

						for (size_t c = nocc; c < nbasis; c++) {
							for (size_t d = nocc; d < nbasis; d++) {
								dik_mat(c - nocc, d - nocc) = dij(i, c, k, d);
								dkj_mat(c - nocc, d - nocc) = dij(k, c, j, d);

								Tik_mat_PNO(c - nocc, d - nocc) = Tij_PNO(i, c,
										k, d);
								Tkj_mat_PNO(c - nocc, d - nocc) = Tij_PNO(k, c,
										j, d);
							}
						}

						Eigen::MatrixXd Sijik = dij_mat.transpose() * dik_mat;

						Eigen::MatrixXd Sijkj = dij_mat.transpose() * dkj_mat;

						//T^(ik)_tilda = S^(ij,ik)*T^(ik)_PNO*S^(ij,ik)transpose

						Eigen::MatrixXd Tik_mat_tilda = Sijik * Tik_mat_PNO
								* Sijik.transpose();
						Eigen::MatrixXd Tkj_mat_tilda = Sijkj * Tkj_mat_PNO
								* Sijkj.transpose();

						summ1 += Ft(j, k) * Tik_mat_tilda(a - nocc, b - nocc)
								+ Tkj_mat_tilda(a - nocc, b - nocc) * Ft(i, k);
					}

					term_4(a - nocc, b - nocc) = summ1;

				}
			}

			R_mat = term_1 + term_3 - term_4;
			//R_mat = term_1 - term_4;

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					Res_PNO(i, a, j, b) = R_mat(a - nocc, b - nocc);
				}
			}

		}
	}

	return Res_PNO;
}

TensorRank4 get_increment_of_amplitude_PNO(TensorRank4 &Res_PNO,
		const TensorRank4 &dij, const Eigen::MatrixXd &Ft,
		const Eigen::MatrixXd &C, const int nocc, const int nbasis) {

	const int nvirt = nbasis - nocc;

	TensorRank4 dT_PNO(nbasis, nbasis, nbasis, nbasis);

	Eigen::MatrixXd F_virt(nvirt, nvirt);
	Eigen::MatrixXd C_virt(nvirt, nvirt);

	for (size_t a = nocc; a < nbasis; a++) {
		for (size_t b = nocc; b < nbasis; b++) {
			F_virt(a - nocc, b - nocc) = Ft(a, b);
			C_virt(a - nocc, b - nocc) = C(a, b);
		}
	}

	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {
			Eigen::MatrixXd Kij_mat(nvirt, nvirt);
			Kij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::MatrixXd dij_mat(nvirt, nvirt);
			dij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					dij_mat(a - nocc, b - nocc) = dij(i, a, j, b);
				}
			}

			Eigen::MatrixXd F_virt_PNO = dij_mat.transpose() * F_virt * dij_mat;
			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolverF_virt_PNO(
					F_virt_PNO);
			Eigen::VectorXd e_virtual_PNO = eigensolverF_virt_PNO.eigenvalues();

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					/*
					 dT_PNO(i, a, j, b) = -Res_PNO(i, a, j, b)
					 / (e_virtual_PNO(a - nocc) + e_virtual_PNO(b - nocc)
					 - Ft(i, i) - Ft(j, j));
					 */
					dT_PNO(i, a, j, b) = -Res_PNO(i, a, j, b)
							/ (F_virt_PNO(a - nocc, a - nocc)
									+ F_virt_PNO(b - nocc, b - nocc) - Ft(i, i)
									- Ft(j, j));

					//std::cout << dT_PNO(i, a, j, b) << std::endl;
				}
			}

		}
	}
	return dT_PNO;
}

void MP2_linear_system_solver_PNO(const TensorRank4 &orb,
		const Eigen::MatrixXd &F, const Eigen::MatrixXd &C,
		const TensorRank4 &Tij, const double nocc, const double nbasis,
		TensorRank4 &dij, TensorRank4 &aij) {

	const int nvirt = nbasis - nocc;

	Eigen::MatrixXd Ft = C.transpose() * F * C;

	TensorRank4 Tij_PNO = transformation_of_doubles_amplitudes_from_MO_into_PNO(
			Tij, dij, nocc, nbasis);

	int count = 0;
	while (true) {

		TensorRank4 Res_PNO = get_residual_PNO(orb, Tij_PNO, dij, Ft, C, nocc,
				nbasis);
		TensorRank4 dT_PNO = get_increment_of_amplitude_PNO(Res_PNO, dij, Ft, C,
				nocc, nbasis);

		//updating new amplitudes as Tij_n = Tij_n + dT
		Tij_PNO = get_new_aplitudes(Tij_PNO, dT_PNO, nocc, nbasis);
		double max = max_abs_Res(Res_PNO, nocc, nbasis);
		std::cout << max << std::endl;

		count++;
		if (max < 1e-7)
			break;
	}

	//transformation of orbitals into PNO basis
	TensorRank4 orb_PNO(nbasis, nbasis, nbasis, nbasis);
	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {
			Eigen::MatrixXd K_mat(nvirt, nvirt);
			K_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd dij_mat(nvirt, nvirt);
			dij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					dij_mat(a - nocc, b - nocc) = dij(i, a, j, b);
					K_mat(a - nocc, b - nocc) = orb(i, a, j, b);
				}
			}

			Eigen::MatrixXd K_mat_PNO = dij_mat.transpose() * K_mat * dij_mat;

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					orb_PNO(i, a, j, b) = K_mat_PNO(a - nocc, b - nocc);
				}
			}

		}
	}

	double E_MP2 = 0;
	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {

			double summ = 0.0;
			double E_MP2_pair = 0.0;
			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {

					summ += orb_PNO(i, a, j, b)
							* (2.0 * Tij_PNO(i, a, j, b) - Tij_PNO(i, b, j, a));
				}
			}
			E_MP2_pair = summ;

			E_MP2 += E_MP2_pair;
		}
	}

	std::cout << "PNO MP2 energy linear systems 2" << std::endl;
	std::cout << E_MP2 << std::endl;
	//std::cout << count << std::endl;
}

TensorRank4 get_residual_truncated_PNO(const TensorRank4 &orb,
		const TensorRank4 Tij_PNO, TensorRank4 &dij, TensorRank4 &aij,
		Eigen::VectorXd &PNOdim, const Eigen::MatrixXd &Ft,
		const Eigen::MatrixXd &C, const int nocc, const int nbasis,
		const double TCutPNO) {

	TensorRank4 Res_PNO(nbasis, nbasis, nbasis, nbasis);

	const int nvirt = nbasis - nocc;

	Eigen::MatrixXd F_virt(nvirt, nvirt);

	for (size_t a = nocc; a < nbasis; a++) {
		for (size_t b = nocc; b < nbasis; b++) {
			F_virt(a - nocc, b - nocc) = Ft(a, b);
		}
	}

	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {

			Eigen::MatrixXd Tij_mat_PNO(nvirt, nvirt);
			Tij_mat_PNO = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd Kij_mat(nvirt, nvirt);
			Kij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::MatrixXd dij_mat(nvirt, nvirt);
			dij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::MatrixXd aij_mat(nvirt, nvirt);
			aij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::VectorXd aij_vect(nvirt);
			aij_vect = Eigen::VectorXd::Zero(nvirt);

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {

					Kij_mat(a - nocc, b - nocc) = orb(i, a, j, b);
					dij_mat(a - nocc, b - nocc) = dij(i, a, j, b);
					aij_mat(a - nocc, b - nocc) = aij(i, a, j, b);
					Tij_mat_PNO(a - nocc, b - nocc) = Tij_PNO(i, a, j, b);
				}
			}

			for (size_t a = nocc; a < nbasis; a++) {
				aij_vect(a - nocc) = aij_mat(a - nocc, a - nocc);
			}

			//std::cout << "PNO occupation number for pair i,j = " << i << ","
			//<< j << std::endl;
			//std::cout << aij_vect << std::endl;
			//std::cout << dij_mat << std::endl;

			//MP2-PNO energy after PNO truncation for ij pair
			//Truncation criteria is TCutPNO < 1e-7

			int dimPNO = 0;

			//counting a number of PNO's for a given pair
			if (TCutPNO == 0) {
				dimPNO = nvirt;
			} else {
				for (size_t p = 0; p < nvirt; p++) {
					if (aij_vect(p) > TCutPNO) {
						dimPNO += 1;
					}
				}
			}

			//aij_vec_T & dij_mat_T are PNO occupation number and
			//coefficients after truncation for given ij pair
			Eigen::MatrixXd dij_mat_T1(nvirt, dimPNO);
			dij_mat_T1 = Eigen::MatrixXd::Zero(nvirt, dimPNO);

			Eigen::MatrixXd dij_mat_T(nvirt, dimPNO + 1);
			dij_mat_T = Eigen::MatrixXd::Zero(nvirt, dimPNO + 1);

			Eigen::VectorXd aij_vect_T(dimPNO);
			aij_vect_T = Eigen::VectorXd::Zero(dimPNO);

			Eigen::MatrixXd Tij_mat_PNO_T(dimPNO, dimPNO);
			Tij_mat_PNO_T = Eigen::MatrixXd::Zero(dimPNO, dimPNO);

			for (size_t p = 0; p < dimPNO; p++) {
				aij_vect_T(p) = aij_vect(nvirt - dimPNO + p);

				for (size_t a = 0; a < nvirt; a++) {
					dij_mat_T1(a, p) = dij_mat(a, nvirt - dimPNO + p);
					//maybe this will not work
				}
				for (size_t q = 0; q < dimPNO; q++) {
					Tij_mat_PNO_T(p, q) = Tij_mat_PNO(nvirt - dimPNO + p,
							nvirt - dimPNO + q);
				}
			}

			Eigen::MatrixXd Res_mat_PNO_T(dimPNO, dimPNO);
			Res_mat_PNO_T = Eigen::MatrixXd::Zero(dimPNO, dimPNO);

			if (dij_mat_T1.cols() != 0) {
				dij_mat_T = dij_mat_T1;

				Eigen::MatrixXd Kij_mat_PNO_T = dij_mat_T.transpose() * Kij_mat
						* dij_mat_T;
				Eigen::MatrixXd S_PNO_T = dij_mat_T.transpose() * dij_mat_T;
				Eigen::MatrixXd F_virt_PNO_T = dij_mat_T.transpose() * F_virt
						* dij_mat_T;

				PNOdim(i * nocc + j) = dimPNO;
				//std::cout << PNOdim << std::endl;

				Eigen::MatrixXd Tint_T(dimPNO, dimPNO);
				Tint_T = Eigen::MatrixXd::Zero(dimPNO, dimPNO);

				for (size_t a = nocc; a < nocc + dimPNO; a++) {
					for (size_t b = nocc; b < nocc + dimPNO; b++) {
						double summ = 0.0;
						for (size_t k = 0; k < nocc; k++) {
							summ += Ft(i, k) * Tij_PNO(k, a, j, b)
									+ Ft(k, j) * Tij_PNO(i, a, k, b);
						}
						Tint_T(a - nocc, b - nocc) = summ;
					}
				}

				//std::cout << Kij_mat_PNO_T << std::endl;
				//std::cout << "Overlap matrix of truncated PNO " << i << ","
				//				<< j << std::endl;
				//std::cout << S_PNO_T << std::endl;
				//std::cout << F_virt_PNO_T << std::endl;
				//std::cout << Tint_T << std::endl;

				//std::cout << dij_mat_T << std::endl;
				Res_mat_PNO_T = Kij_mat_PNO_T
						+ F_virt_PNO_T * Tij_mat_PNO_T * S_PNO_T
						+ S_PNO_T * Tij_mat_PNO_T * F_virt_PNO_T
						- S_PNO_T * Tint_T * S_PNO_T;
				//std::cout << Res_mat_PNO_T << std::endl;

				for (size_t a = nocc; a < nocc + dimPNO; a++) {
					for (size_t b = nocc; b < nocc + dimPNO; b++) {
						Res_PNO(i, a, j, b) = Res_mat_PNO_T(a - nocc, b - nocc);

					}
				}

				//std::cout << Res_mat_PNO_T << std::endl;

			}
		}
	}

	//std::cout << PNOdim << std::endl;
	return Res_PNO;
}

TensorRank4 get_increment_of_amplitude_truncated_PNO(TensorRank4 &Res_PNO,
		const TensorRank4 &dij, const TensorRank4 &aij, Eigen::VectorXd PNOdim,
		const Eigen::MatrixXd &Ft, const Eigen::MatrixXd &C, const int nocc,
		const int nbasis, const double TCutPNO) {

	TensorRank4 dT_PNO_T(nbasis, nbasis, nbasis, nbasis);

	const int nvirt = nbasis - nocc;

	Eigen::MatrixXd F_virt(nvirt, nvirt);

	for (size_t a = nocc; a < nbasis; a++) {
		for (size_t b = nocc; b < nbasis; b++) {
			F_virt(a - nocc, b - nocc) = Ft(a, b);
		}
	}

	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {

			Eigen::MatrixXd Tij_mat_PNO(nvirt, nvirt);
			Tij_mat_PNO = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd Kij_mat(nvirt, nvirt);
			Kij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::MatrixXd dij_mat(nvirt, nvirt);
			dij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::MatrixXd aij_mat(nvirt, nvirt);
			aij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::VectorXd aij_vect(nvirt);
			aij_vect = Eigen::VectorXd::Zero(nvirt);

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {

					dij_mat(a - nocc, b - nocc) = dij(i, a, j, b);
					aij_mat(a - nocc, b - nocc) = aij(i, a, j, b);
				}
			}

			for (size_t a = nocc; a < nbasis; a++) {
				aij_vect(a - nocc) = aij_mat(a - nocc, a - nocc);
			}

			//std::cout << "PNO occupation number for pair i,j = " << i << ","
			//	<< j << std::endl;
			//std::cout << aij_vect << std::endl;
			//std::cout << dij_mat << std::endl;

			//MP2-PNO energy after PNO truncation for ij pair
			//Truncation criteria is TCutPNO < 1e-7

			int dimPNO = 0;

			//counting a number of PNO's for a given pair
			if (TCutPNO == 0) {
				dimPNO = nvirt;
			} else {
				for (size_t p = 0; p < nvirt; p++) {
					if (aij_vect(p) > TCutPNO) {
						dimPNO += 1;
					}
				}
			}

			//aij_vec_T & dij_mat_T are PNO occupation number and
			//coefficients after truncation for given ij pair
			Eigen::MatrixXd dij_mat_T1(nvirt, dimPNO);
			dij_mat_T1 = Eigen::MatrixXd::Zero(nvirt, dimPNO);

			Eigen::MatrixXd dij_mat_T(nvirt, dimPNO + 1);
			dij_mat_T = Eigen::MatrixXd::Zero(nvirt, dimPNO + 1);

			Eigen::VectorXd aij_vect_T(dimPNO);
			aij_vect_T = Eigen::VectorXd::Zero(dimPNO);

			for (size_t p = 0; p < dimPNO; p++) {
				aij_vect_T(p) = aij_vect(nvirt - dimPNO + p);

				for (size_t a = 0; a < nvirt; a++) {
					dij_mat_T1(a, p) = dij_mat(a, nvirt - dimPNO + p);
				}
			}

			//std::cout << "PNO occupation number for pair i,j = " << i << ","
			//	<< j << std::endl;

			if (dij_mat_T1.cols() != 0) {

				dij_mat_T = dij_mat_T1;

				Eigen::MatrixXd dT_mat_PNO_T(dimPNO, dimPNO);
				dT_mat_PNO_T = Eigen::MatrixXd::Zero(dimPNO, dimPNO);

				Eigen::MatrixXd F_virt_PNO_T = dij_mat_T.transpose() * F_virt
						* dij_mat_T;
				Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolverF_virt_PNO(
						F_virt_PNO_T);
				Eigen::VectorXd e_virtual_PNO =
						eigensolverF_virt_PNO.eigenvalues();

				for (size_t a = nocc; a < nocc + dimPNO; a++) {
					for (size_t b = nocc; b < nocc + dimPNO; b++) {
						dT_mat_PNO_T(a - nocc, b - nocc) = -Res_PNO(i, a, j, b)
								/ (F_virt_PNO_T(a - nocc, a - nocc)
										+ F_virt_PNO_T(b - nocc, b - nocc)
										- Ft(i, i) - Ft(j, j));
						dT_PNO_T(i, a, j, b) = dT_mat_PNO_T(a - nocc, b - nocc);
					}
				}

				//std::cout << dT_mat_PNO_T << std::endl;
			}

		}
	}
	return dT_PNO_T;
}

TensorRank4 get_new_aplitudes_truncated_PNO(TensorRank4 &Tij_PNO_T,
		const TensorRank4 &dT_PNO_T, const TensorRank4 &dij,
		const TensorRank4 &aij, const int nocc, const int nbasis,
		const double TCutPNO) {

	const int nvirt = nbasis - nocc;

	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {

			Eigen::MatrixXd Tij_mat_PNO(nvirt, nvirt);
			Tij_mat_PNO = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd Kij_mat(nvirt, nvirt);
			Kij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::MatrixXd dij_mat(nvirt, nvirt);
			dij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::MatrixXd aij_mat(nvirt, nvirt);
			aij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::VectorXd aij_vect(nvirt);
			aij_vect = Eigen::VectorXd::Zero(nvirt);

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {

					dij_mat(a - nocc, b - nocc) = dij(i, a, j, b);
					aij_mat(a - nocc, b - nocc) = aij(i, a, j, b);
				}
			}

			for (size_t a = nocc; a < nbasis; a++) {
				aij_vect(a - nocc) = aij_mat(a - nocc, a - nocc);
			}

			//std::cout << "PNO occupation number for pair i,j = " << i << ","
			//	<< j << std::endl;
			//std::cout << aij_vect << std::endl;
			//std::cout << dij_mat << std::endl;

			//MP2-PNO energy after PNO truncation for ij pair
			//Truncation criteria is TCutPNO < 1e-7

			int dimPNO = 0;

			//counting a number of PNO's for a given pair
			if (TCutPNO == 0) {
				dimPNO = nvirt;
			} else {
				for (size_t p = 0; p < nvirt; p++) {
					if (aij_vect(p) > TCutPNO) {
						dimPNO += 1;
					}
				}
			}

			//aij_vec_T & dij_mat_T are PNO occupation number and
			//coefficients after truncation for given ij pair
			Eigen::MatrixXd dij_mat_T1(nvirt, dimPNO);
			dij_mat_T1 = Eigen::MatrixXd::Zero(nvirt, dimPNO);

			Eigen::MatrixXd dij_mat_T(nvirt, dimPNO + 1);
			dij_mat_T = Eigen::MatrixXd::Zero(nvirt, dimPNO + 1);

			Eigen::VectorXd aij_vect_T(dimPNO);
			aij_vect_T = Eigen::VectorXd::Zero(dimPNO);

			for (size_t p = 0; p < dimPNO; p++) {
				aij_vect_T(p) = aij_vect(nvirt - dimPNO + p);

				for (size_t a = 0; a < nvirt; a++) {
					dij_mat_T1(a, p) = dij_mat(a, nvirt - dimPNO + p);
				}
			}

			//std::cout << "PNO occupation number for pair i,j = " << i << ","
			//	<< j << std::endl;

			if (dij_mat_T1.cols() != 0) {
				dij_mat_T = dij_mat_T1;
				for (size_t a = nocc; a < nocc + dimPNO; a++) {
					for (size_t b = nocc; b < nocc + dimPNO; b++) {
						Tij_PNO_T(i, a, j, b) = Tij_PNO_T(i, a, j, b)
								+ dT_PNO_T(i, a, j, b);
					}
				}

			}
		}
	}

	return Tij_PNO_T;
}

double get_energy_truncated_PNO(const TensorRank4 &orb, TensorRank4 &Tij_PNO_T,
		TensorRank4 &dij, TensorRank4 &aij, const int nocc, const int nbasis,
		const double TCutPNO) {

	double E_MP2_PNO_T = 0.0;

	const int nvirt = nbasis - nocc;

	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {

			Eigen::MatrixXd Tij_mat_PNO(nvirt, nvirt);
			Tij_mat_PNO = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd Kij_mat(nvirt, nvirt);
			Kij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::MatrixXd dij_mat(nvirt, nvirt);
			dij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::MatrixXd aij_mat(nvirt, nvirt);
			aij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::VectorXd aij_vect(nvirt);
			aij_vect = Eigen::VectorXd::Zero(nvirt);

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {

					dij_mat(a - nocc, b - nocc) = dij(i, a, j, b);
					aij_mat(a - nocc, b - nocc) = aij(i, a, j, b);
					Kij_mat(a - nocc, b - nocc) = orb(i, a, j, b);
				}
			}

			for (size_t a = nocc; a < nbasis; a++) {
				aij_vect(a - nocc) = aij_mat(a - nocc, a - nocc);
			}

			//std::cout << "PNO occupation number for pair i,j = " << i << ","
			//	<< j << std::endl;
			//std::cout << aij_vect << std::endl;
			//std::cout << dij_mat << std::endl;

			//MP2-PNO energy after PNO truncation for ij pair
			//Truncation criteria is TCutPNO < 1e-7

			int dimPNO = 0;

			//counting a number of PNO's for a given pair
			if (TCutPNO == 0) {
				dimPNO = nvirt;
			} else {
				for (size_t p = 0; p < nvirt; p++) {
					if (aij_vect(p) > TCutPNO) {
						dimPNO += 1;
					}
				}
			}

			//aij_vec_T & dij_mat_T are PNO occupation number and
			//coefficients after truncation for given ij pair
			Eigen::MatrixXd dij_mat_T1(nvirt, dimPNO);
			dij_mat_T1 = Eigen::MatrixXd::Zero(nvirt, dimPNO);

			Eigen::MatrixXd dij_mat_T(nvirt, dimPNO + 1);
			dij_mat_T = Eigen::MatrixXd::Zero(nvirt, dimPNO + 1);

			Eigen::VectorXd aij_vect_T(dimPNO);
			aij_vect_T = Eigen::VectorXd::Zero(dimPNO);

			for (size_t p = 0; p < dimPNO; p++) {
				aij_vect_T(p) = aij_vect(nvirt - dimPNO + p);

				for (size_t a = 0; a < nvirt; a++) {
					dij_mat_T1(a, p) = dij_mat(a, nvirt - dimPNO + p);
				}
			}

			double E_MP2_pair = 0.0;
			if (dij_mat_T1.cols() != 0) {
				dij_mat_T = dij_mat_T1;

				Eigen::MatrixXd Kij_PNO_T(dimPNO, dimPNO);
				Kij_PNO_T = Eigen::MatrixXd::Zero(dimPNO, dimPNO);
				//transformation of K_matrix into truncated PNO basis
				Kij_PNO_T = dij_mat_T.transpose() * Kij_mat * dij_mat_T;
				//std::cout << Kij_PNO_T << std::endl;

				for (size_t a = nocc; a < nocc + dimPNO; a++) {
					for (size_t b = nocc; b < nocc + dimPNO; b++) {
						E_MP2_pair += Kij_PNO_T(a - nocc, b - nocc)
								* (2.0 * Tij_PNO_T(i, a, j, b)
										- Tij_PNO_T(i, b, j, a));
						//std::cout << E_MP2_pair << std::endl;
					}
				}
			}

			E_MP2_PNO_T += E_MP2_pair;
		}
	}

	return E_MP2_PNO_T;
}

double max_abs_Res_truncated_PNO(const TensorRank4 &Res_PNO_T,
		const TensorRank4 &dij, const TensorRank4 &aij, const int nocc,
		const int nbasis, const double TCutPNO) {

	double max = 0.0;

	const int nvirt = nbasis - nocc;

	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {

			Eigen::MatrixXd Tij_mat_PNO(nvirt, nvirt);
			Tij_mat_PNO = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd Kij_mat(nvirt, nvirt);
			Kij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::MatrixXd dij_mat(nvirt, nvirt);
			dij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::MatrixXd aij_mat(nvirt, nvirt);
			aij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::VectorXd aij_vect(nvirt);
			aij_vect = Eigen::VectorXd::Zero(nvirt);

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {

					dij_mat(a - nocc, b - nocc) = dij(i, a, j, b);
					aij_mat(a - nocc, b - nocc) = aij(i, a, j, b);
				}
			}

			for (size_t a = nocc; a < nbasis; a++) {
				aij_vect(a - nocc) = aij_mat(a - nocc, a - nocc);
			}

			int dimPNO = 0;

			//counting a number of PNO's for a given pair
			if (TCutPNO == 0) {
				dimPNO = nvirt;
			} else {
				for (size_t p = 0; p < nvirt; p++) {
					if (aij_vect(p) > TCutPNO) {
						dimPNO += 1;
					}
				}
			}

			//aij_vec_T & dij_mat_T are PNO occupation number and
			//coefficients after truncation for given ij pair
			Eigen::MatrixXd dij_mat_T1(nvirt, dimPNO);
			dij_mat_T1 = Eigen::MatrixXd::Zero(nvirt, dimPNO);

			Eigen::MatrixXd dij_mat_T(nvirt, dimPNO + 1);
			dij_mat_T = Eigen::MatrixXd::Zero(nvirt, dimPNO + 1);

			Eigen::VectorXd aij_vect_T(dimPNO);
			aij_vect_T = Eigen::VectorXd::Zero(dimPNO);

			for (size_t p = 0; p < dimPNO; p++) {
				aij_vect_T(p) = aij_vect(nvirt - dimPNO + p);

				for (size_t a = 0; a < nvirt; a++) {
					dij_mat_T1(a, p) = dij_mat(a, nvirt - dimPNO + p);
				}
			}

			if (dij_mat_T1.cols() != 0) {
				dij_mat_T = dij_mat_T1;

				for (size_t a = nocc; a < nocc + dimPNO; a++) {
					for (size_t b = nocc; b < nocc + dimPNO; b++) {
						max = std::max(max, std::abs(Res_PNO_T(i, a, j, b)));
					}
				}

			}
		}
	}
	return max;
}

void MP2_linear_system_solver_truncated_PNO(const TensorRank4 &orb,
		const Eigen::MatrixXd &F, const Eigen::MatrixXd &C,
		const TensorRank4 &Tij, const double nocc, const double nbasis,
		TensorRank4 &dij, TensorRank4 &aij) {

	const int nvirt = nbasis - nocc;

	//truncated tensor
	TensorRank4 dij_T(nbasis, nbasis, nbasis, nbasis);

	Eigen::MatrixXd Ft = C.transpose() * F * C;

	TensorRank4 Tij_PNO = transformation_of_doubles_amplitudes_from_MO_into_PNO(
			Tij, dij, nocc, nbasis);

	double E_MP2_PNO_T;

	double TCutPNO = 1e-6;

	Eigen::VectorXd PNOdim(nocc * nocc);
	PNOdim = Eigen::VectorXd::Zero(nocc * nocc);

	int count = 0;
	while (true) {

		TensorRank4 Res_PNO_T = get_residual_truncated_PNO(orb, Tij_PNO, dij,
				aij, PNOdim, Ft, C, nocc, nbasis, TCutPNO);

		TensorRank4 dT_PNO = get_increment_of_amplitude_truncated_PNO(Res_PNO_T,
				dij, aij, PNOdim, Ft, C, nocc, nbasis, TCutPNO);

		Tij_PNO = get_new_aplitudes_truncated_PNO(Tij_PNO, dT_PNO, dij, aij,
				nocc, nbasis, TCutPNO);

		E_MP2_PNO_T = get_energy_truncated_PNO(orb, Tij_PNO, dij, aij, nocc,
				nbasis, TCutPNO);

		double max = max_abs_Res_truncated_PNO(Res_PNO_T, dij, aij, nocc,
				nbasis, TCutPNO);

		//std::cout << max << std::endl;
		count++;
		if (max < 1e-7)
			break;
	}

	std::cout << "PNO MP2 energy linear solver with truncation" << std::endl;
	std::cout << E_MP2_PNO_T << std::endl;

}

TensorRank4 get_residual_CEPA_doubles(std::string method,
		const TensorRank4 &orb, const TensorRank4 Tij, Eigen::MatrixXd &t,
		const Eigen::MatrixXd &Ft, const int nocc, const int nbasis,
		double E_CEPA, Eigen::MatrixXd e_d, Eigen::VectorXd e_s) {

	TensorRank4 Res_doubles(nbasis, nbasis, nbasis, nbasis);

	const int nvirt = nbasis - nocc;

	Eigen::MatrixXd F_virt(nvirt, nvirt);

	for (size_t a = nocc; a < nbasis; a++) {
		for (size_t b = nocc; b < nbasis; b++) {
			F_virt(a - nocc, b - nocc) = Ft(a, b);
		}
	}

	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {

			Eigen::MatrixXd R_mat(nvirt, nvirt);
			R_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd K_mat(nvirt, nvirt);
			K_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::MatrixXd K_virt(nvirt, nvirt);
			K_virt = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::MatrixXd Tij_mat(nvirt, nvirt);
			Tij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::MatrixXd Tij_virt(nvirt, nvirt);
			Tij_virt = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd term_2(nvirt, nvirt);
			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {

					double summ_virt = 0.0;
					for (size_t c = nocc; c < nbasis; c++) {
						for (size_t d = nocc; d < nbasis; d++) {
							K_virt(c - nocc, d - nocc) = orb(a, c, b, d);
							Tij_virt(c - nocc, d - nocc) = Tij(i, c, j, d);
							summ_virt += K_virt(c - nocc, d - nocc)
									* Tij_virt(c - nocc, d - nocc);
						}
					}

					term_2(a - nocc, b - nocc) = summ_virt;
					K_mat(a - nocc, b - nocc) = orb(i, a, j, b);
					Tij_mat(a - nocc, b - nocc) = Tij(i, a, j, b);

				}
			}

			Eigen::MatrixXd term_1 = K_mat;

			Eigen::MatrixXd term_3 = F_virt * Tij_mat + Tij_mat * F_virt;

			Eigen::MatrixXd term_4(nvirt, nvirt);
			term_4 = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd term_5(nvirt, nvirt);
			term_5 = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd term_6(nvirt, nvirt);
			term_6 = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd term_7(nvirt, nvirt);
			term_7 = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd term_8(nvirt, nvirt);
			term_8 = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd term_9(nvirt, nvirt);
			term_9 = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd term_10(nvirt, nvirt);
			term_10 = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd term_11(nvirt, nvirt);
			term_11 = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd Kia(nvirt, nvirt);
			Kia = Eigen::MatrixXd::Zero(nvirt, nvirt);

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {

					double summ1 = 0.0;
					for (size_t k = 0; k < nocc; k++) {
						summ1 += Ft(j, k) * Tij(i, a, k, b)
								+ Ft(i, k) * Tij(k, a, j, b);
					}

					term_4(a - nocc, b - nocc) = summ1;

					double summ2 = 0.0;
					for (size_t k = 0; k < nocc; k++) {
						for (size_t l = 0; l < nocc; l++) {
							//for (size_t k = nocc; k < nbasis; k++) {
							//for (size_t l = nocc; l < nbasis; l++) {
							summ2 += orb(i, k, j, l) * Tij(k, a, l, b);
						}
					}
					term_5(a - nocc, b - nocc) = summ2;

					double summ3 = 0.0;
					for (size_t k = 0; k < nocc; k++) {
						for (size_t e = nocc; e < nbasis; e++) {

							summ3 += (2 * Tij(i, a, k, e) - Tij(i, e, k, a))
									* (orb(k, e, j, b) - 0.5 * orb(k, j, e, b))
									+ (orb(i, a, k, e) - 0.5 * orb(i, k, a, e))
											* (2 * Tij(k, e, j, b)
													- Tij(k, b, j, e));
						}
					}

					term_6(a - nocc, b - nocc) = summ3;

					double summ4 = 0.0;
					for (size_t k = 0; k < nocc; k++) {
						for (size_t e = nocc; e < nbasis; e++) {

							summ4 += 0.5 * Tij(i, e, k, a) * orb(j, k, b, e)
									+ 0.5 * orb(i, k, a, e) * Tij(k, b, j, e)
									+ orb(j, k, a, e) * Tij(i, e, k, b)
									+ Tij(k, a, j, e) * orb(i, k, b, e);
							//summ4 += 0.5 * Tij(i, e, k, a) * orb(j, k, b, e);
						}
					}

					term_7(a - nocc, b - nocc) = summ4;

					term_8(a - nocc, b - nocc) = t(i, a - nocc) * Ft(j, b)
							+ t(j, b - nocc) * Ft(i, a);

					double summ5 = 0.0;
					//wrong sign
					for (size_t k = 0; k < nocc; k++) {
						//or other way around for t
						summ5 += orb(i, k, j, b) * t(k, a - nocc)
								+ orb(j, k, i, a) * t(k, b - nocc);
					}
					term_9(a - nocc, b - nocc) = summ5;

					double summ6 = 0.0;
					//wrong sign
					for (size_t e = nocc; e < nbasis; e++) {
						summ6 += orb(j, b, a, e) * t(i, e - nocc)
								+ orb(i, a, e, b) * t(j, e - nocc);

					}
					term_10(a - nocc, b - nocc) = summ6;

					double summ7 = 0.0;
					for (size_t k = 0; k < nocc; k++) {
						summ7 += e_d(i, k) + e_d(j, k);
					}

					//CEPA/0 method
					if (method == "CEPA/0") {
						term_11(a - nocc, b - nocc) = 0.0;
					}

					//CISD mehod
					else if (method == "CISD") {
						term_11(a - nocc, b - nocc) = E_CEPA * Tij(i, a, j, b);
					}

					//CEPA/1
					else if (method == "CEPA/1") {
						term_11(a - nocc, b - nocc) = (0.25 * summ7)
								* Tij(i, a, j, b);
					}

					//NCEPA/2
					else if (method == "NCEPA/2") {
						term_11(a - nocc, b - nocc) = e_d(i, j);
					}

					//ACPF, ACPF/2, NACPF
					else if (method == "ACPF" || method == "ACPF/2"
							|| method == "NACPF") {
						term_11(a - nocc, b - nocc) = 1.0 / nocc * E_CEPA
								* Tij(i, a, j, b);
					}

					//AQCC
					else if (method == "AQCC") {
						term_11(a - nocc, b - nocc) = (1.0
								- (2.0 * nocc - 3.0) * (2.0 * nocc - 2.0)
										/ (2.0 * nocc * (2.0 * nocc - 1.0)))
								* E_CEPA * Tij(i, a, j, b);
					}

				}

			}

			R_mat = term_1 + term_2 + term_3 - term_4 + term_5 + term_6 - term_7
					+ term_8 - term_9 + term_10 - term_11;

			//R_mat = term_1 + term_2 + term_3 - term_4 + term_5 + term_6 - term_7
			//	- term_9 + term_10;
			//std::cout << "Res for pair i,j = " << i << "," << j << std::endl;
			//std::cout << term_10 << std::endl;
			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					Res_doubles(i, a, j, b) = R_mat(a - nocc, b - nocc);
				}
			}

		}

	}

	return Res_doubles;
}

Eigen::MatrixXd get_residual_CEPA_singles(std::string method,
		const TensorRank4 &orb, const TensorRank4 Tij, Eigen::MatrixXd &t,
		const Eigen::MatrixXd &Ft, const int nocc, const int nbasis, int count,
		double E_CEPA, Eigen::MatrixXd e_d) {

	Eigen::MatrixXd Res_singles(nbasis, nbasis);

	const int nvirt = nbasis - nocc;

	Eigen::MatrixXd F_virt(nvirt, nvirt);

	for (size_t a = nocc; a < nbasis; a++) {
		for (size_t b = nocc; b < nbasis; b++) {
			F_virt(a - nocc, b - nocc) = Ft(a, b);
		}
	}

	Eigen::MatrixXd K_mat(nocc, nvirt);
	Eigen::MatrixXd Tij_mat(nvirt, nvirt);

	Eigen::MatrixXd term_1(nocc, nvirt);
	term_1 = Eigen::MatrixXd::Zero(nocc, nvirt);

	Eigen::MatrixXd term_2(nocc, nvirt);
	term_2 = Eigen::MatrixXd::Zero(nocc, nvirt);

	Eigen::MatrixXd term_3(nocc, nvirt);
	term_3 = Eigen::MatrixXd::Zero(nocc, nvirt);

	Eigen::MatrixXd term_4(nocc, nvirt);
	term_4 = Eigen::MatrixXd::Zero(nocc, nvirt);

	Eigen::MatrixXd term_5(nocc, nvirt);
	term_5 = Eigen::MatrixXd::Zero(nocc, nvirt);

	Eigen::MatrixXd term_6(nocc, nvirt);
	term_6 = Eigen::MatrixXd::Zero(nocc, nvirt);

	Eigen::MatrixXd term_7(nocc, nvirt);
	term_7 = Eigen::MatrixXd::Zero(nocc, nvirt);

	Eigen::MatrixXd term_8(nocc, nvirt);
	term_8 = Eigen::MatrixXd::Zero(nocc, nvirt);

	Eigen::VectorXd Fv(nvirt);
	Fv = Eigen::VectorXd::Zero(nvirt);

	Eigen::VectorXd t_vec(nvirt);
	t_vec = Eigen::VectorXd::Zero(nvirt);

	for (size_t i = 0; i < nocc; i++) {
		for (size_t a = nocc; a < nbasis; a++) {

			term_1(i, a - nocc) = Ft(i, a);
		}

		//or other way around
		//Eigen::MatrixXd term_2 =  t.transpose()*Fv;

		for (size_t a = nocc; a < nbasis; a++) {
			double summ0 = 0.0;

			for (size_t e = nocc; e < nbasis; e++) {
				summ0 += Ft(a, e) * t(i, e - nocc);
			}
			term_2(i, a - nocc) = summ0;
		}

		for (size_t a = nocc; a < nbasis; a++) {

			double summ1 = 0.0;

			for (size_t j = 0; j < nocc; j++) {
				summ1 += Ft(i, j) * t(j, a - nocc);
			}

			term_3(i, a - nocc) = summ1;
		}

		for (size_t a = nocc; a < nbasis; a++) {

			double summ2 = 0.0;
			//wrong sign
			for (size_t j = 0; j < nocc; j++) {
				for (size_t k = 0; k < nocc; k++) {
					for (size_t b = nocc; b < nbasis; b++) {
						summ2 += (2 * orb(i, j, k, b) - orb(i, k, j, b))
								* Tij(k, b, j, a);
					}
				}
			}

			term_4(i, a - nocc) = summ2;
		}

		for (size_t a = nocc; a < nbasis; a++) {

			double summ3 = 0.0;
			for (size_t j = 0; j < nocc; j++) {
				for (size_t e = nocc; e < nbasis; e++) {

					summ3 += (2 * orb(i, a, j, e) - orb(i, j, a, e))
							* t(j, e - nocc);
				}
			}

			term_5(i, a - nocc) = summ3;
		}

		for (size_t a = nocc; a < nbasis; a++) {

			double summ4 = 0.0;
			for (size_t j = 0; j < nocc; j++) {
				for (size_t e = nocc; e < nbasis; e++) {

					summ4 += Ft(j, e) * (2 * Tij(i, a, j, e) - Tij(i, e, j, a));
				}
			}

			term_6(i, a - nocc) = summ4;
		}

		for (size_t a = nocc; a < nbasis; a++) {
			double summ5 = 0.0;
//wrong sign
			for (size_t j = 0; j < nocc; j++) {
				for (size_t e = nocc; e < nbasis; e++) {
					for (size_t f = nocc; f < nbasis; f++) {
						summ5 += (2 * orb(j, e, a, f) - orb(j, f, a, e))
								* Tij(i, f, j, e);
					}
				}
			}
			term_7(i, a - nocc) = summ5;

		}

		for (size_t a = nocc; a < nbasis; a++) {
			//CEPA/0
			if (method == "CEPA/0") {
				term_8(i, a - nocc) = 0.0;
			}

			//CISD
			else if (method == "CISD") {
				term_8(i, a - nocc) = E_CEPA * t(i, a - nocc);
			}

			//CEPA/1
			else if (method == "CEPA/1") {
				double summ5 = 0.0;
				for (size_t k = 0; k < nocc; k++) {
					summ5 += e_d(i, k);
				}
				term_8(i, a - nocc) = (e_d(i, i) + summ5) * t(i, a - nocc);
			}

			//NCEPA/2
			else if (method == "NCEPA/2") {
				term_8(i, a - nocc) = 2 * e_d(i, i);
			}

			//ACPF
			else if (method == "ACPF") {
				term_8(i, a - nocc) = 1.0 / nocc * E_CEPA * t(i, a - nocc);
			}

			//NACPF
			else if (method == "NACPF") {
				term_8(i, a - nocc) = 2.0 / nocc * E_CEPA * t(i, a - nocc);
			}

			//AQCC, ACPF/2
			else if (method == "AQCC" || method == "ACPF/2") {
				term_8(i, a - nocc) = (1.0
						- (2.0 * nocc - 3.0) * (2.0 * nocc - 2.0)
								/ (2.0 * nocc * (2.0 * nocc - 1.0))) * E_CEPA
						* t(i, a - nocc);
			}

		}
	}

	Res_singles = 1.0
			* (term_1 + term_2 - term_3 - term_4 + term_5 + term_6 + term_7
					- term_8);

	//Res_singles = term_1 + term_2 - term_3 - term_4 + term_5 + term_6 + term_7;

	return Res_singles;
}

TensorRank4 get_amplitude_increment_doubles(const TensorRank4 &Res,
		const Eigen::MatrixXd &Ft, const int nocc, const int nbasis) {
	TensorRank4 dT(nbasis, nbasis, nbasis, nbasis);

//std::cout << Ft << std::endl;
	const int nvirt = nbasis - nocc;
	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {
			Eigen::MatrixXd dT_mat(nvirt, nvirt);
			dT_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd Res_mat(nvirt, nvirt);
			Res_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {

					dT(i, a, j, b) = -Res(i, a, j, b)
							/ (Ft(a, a) + Ft(b, b) - Ft(i, i) - Ft(j, j));
					dT_mat(a - nocc, b - nocc) = dT(i, a, j, b);
					Res_mat(a - nocc, b - nocc) = Res(i, a, j, b);
				}
			}
			//std::cout << "dT_mat for pair i,j = " << i << ","
			//									<< j << std::endl;
			//std::cout << dT_mat << std::endl;
		}
	}

	return dT;
}

Eigen::MatrixXd get_amplitude_increment_singles(const Eigen::MatrixXd &Res,
		const Eigen::MatrixXd &Ft, const int nocc, const int nbasis) {

	const int nvirt = nbasis - nocc;
	Eigen::MatrixXd dt(nocc, nbasis - nocc);

//std::cout << Ft << std::endl;

//std::cout << dt << std::endl;
//std::cout << Res << std::endl;
	for (size_t i = 0; i < nocc; i++) {
		for (size_t a = 0; a < nvirt; a++) {
			dt(i, a) = -Res(i, a) / (Ft(a + nocc, a + nocc) - Ft(i, i));
		}
	}

	return dt;
}

TensorRank4 get_new_amplitudes_doubles(TensorRank4 &Tij, const TensorRank4 &dT,
		const int nocc, const int nbasis) {

	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {

			//Eigen::MatrixXd dT_mat(nvirt, nvirt);

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					Tij(i, a, j, b) = Tij(i, a, j, b) + dT(i, a, j, b);
				}
			}

		}
	}

	return Tij;
}

Eigen::MatrixXd get_new_amplitudes_singles(Eigen::MatrixXd &t,
		Eigen::MatrixXd &dt, const int nocc, const int nbasis) {

	int nvirt = nbasis - nocc;

	for (size_t i = 0; i < nocc; i++) {
		for (size_t a = 0; a < nvirt; a++) {

			t(i, a) = t(i, a) + dt(i, a);
		}
	}

//std::cout << t << std::endl;
	return t;
}

double get_energies_CEPA(const TensorRank4 &orb, const TensorRank4 &Tij,
		const int nocc, const int nbasis) {

	double E_CEPA = 0.0;
	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {

			double E_CEPA_pair = 0.0;
			double summ = 0;
			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {

					summ += orb(i, a, j, b)
							* (2.0 * Tij(i, a, j, b) - Tij(i, b, j, a));
				}
			}
			E_CEPA_pair = summ;

			E_CEPA += E_CEPA_pair;
		}
	}

	return E_CEPA;

}

Eigen::MatrixXd get_doubles_excitations_energies_CEPA(const TensorRank4 &orb,
		const TensorRank4 &Tij, const int nocc, const int nbasis) {

	Eigen::MatrixXd e_d(nocc, nocc);
	e_d = Eigen::MatrixXd::Zero(nocc, nocc);

	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {

			double E_CEPA_pair = 0.0;
			double summ = 0.0;
			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {

					summ += orb(i, a, j, b)
							* (2.0 * Tij(i, a, j, b) - Tij(i, b, j, a));
				}
			}
			e_d(i, j) = summ;

		}
	}

	return e_d;

}

Eigen::VectorXd get_singles_excitations_energies_CEPA(const TensorRank4 &orb,
		const Eigen::MatrixXd &t, const int nocc, const int nbasis) {

	Eigen::VectorXd e_s(nocc);
	e_s = Eigen::VectorXd::Zero(nocc);

	for (size_t i = 0; i < nocc; i++) {

		double summ = 0.0;
		for (size_t a = nocc; a < nbasis; a++) {

			summ += orb(i, a, i, a) * t(i, a - nocc);
		}

		e_s(i) = summ;

	}

	return e_s;

}

double max_abs_Res_doubles(TensorRank4 &Res, const int nocc, const int nbasis) {

	double max = 0.0;

	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {
			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					max = std::max(max, std::abs(Res(i, a, j, b)));
				}
			}
		}
	}
	return max;
}

double max_abs_Res_singles(Eigen::MatrixXd &Res, const int nocc,
		const int nbasis) {

	double max = 0.0;
	int nvirt = nbasis - nocc;

	for (size_t i = 0; i < nocc; i++) {
		for (size_t a = 0; a < nvirt; a++) {
			max = std::max(max, std::abs(Res(i, a)));
		}
	}
	return max;
}

void CEPA_linear_solver(const TensorRank4 &orb, const Eigen::MatrixXd &F,
		const Eigen::MatrixXd &C, TensorRank4 Tij, const int nocc,
		const int nbasis) {

	Eigen::MatrixXd Ft = C.transpose() * F * C;
//std::cout << Ft << std::endl;

//TensorRank4 Tij(nbasis, nbasis, nbasis, nbasis);

	double nvirt = nbasis - nocc;

	Eigen::MatrixXd t(nocc, nbasis - nocc);
//t = Eigen::MatrixXd::Zero(nocc, nbasis - nocc);

	Eigen::MatrixXd e_d(nocc, nocc);
	e_d = Eigen::MatrixXd::Zero(nocc, nocc);

	Eigen::VectorXd e_s(nocc);
//e_s = Eigen::VectorXd::Zero(nocc);

	for (size_t i = 0; i < nocc; i++) {
		for (size_t a = nocc; a < nbasis; a++) {
			t(i, a - nocc) = 0.0;
		}
	}

	double E_CEPA = 0.0;
	int count = 0;

	std::string method;
	//methods of choice: CEPA/0, CEPA/1, CISD, NCEPA/2, ACPF, ACPF/2, NACPF, AQCC
	method = "CEPA/0";

	std::cout << "\n" << std::endl;
	while (true) {

		TensorRank4 Res_doubles = get_residual_CEPA_doubles(method, orb, Tij, t,
				Ft, nocc, nbasis, E_CEPA, e_d, e_s);

		Eigen::MatrixXd Res_singles = get_residual_CEPA_singles(method, orb,
				Tij, t, Ft, nocc, nbasis, count, E_CEPA, e_d);

		TensorRank4 dT = get_amplitude_increment_doubles(Res_doubles, Ft, nocc,
				nbasis);

		Eigen::MatrixXd dt = get_amplitude_increment_singles(Res_singles, Ft,
				nocc, nbasis);

		Tij = get_new_amplitudes_doubles(Tij, dT, nocc, nbasis);

		t = get_new_amplitudes_singles(t, dt, nocc, nbasis);

		E_CEPA = get_energies_CEPA(orb, Tij, nocc, nbasis);

		e_d = get_doubles_excitations_energies_CEPA(orb, Tij, nocc, nbasis);

		e_s = get_singles_excitations_energies_CEPA(orb, t, nocc, nbasis);

		//std::cout << E_CEPA << std::endl;

		double max_d = max_abs_Res_doubles(Res_doubles, nocc, nbasis);
		double max_s = max_abs_Res_singles(Res_singles, nocc, nbasis);

		count++;

		//std::cout << "Correlation Energy: " << E_CEPA << "  Residual_double: " << max_d << std::endl;
		if (max_d < 1e-8 && max_s < 1e-8)
			//	if (max_d < 1e-8)
			break;

	}

	std::cout << method << "\t" << "energy" << std::endl;
	std::cout << E_CEPA << std::endl;
}

TensorRank4 orbital_transformation_to_PNO(const TensorRank4 &orb,
		const TensorRank4 &dij, const int nbasis, const int nocc) {

	TensorRank4 orb_PNO(nbasis, nbasis, nbasis, nbasis);

	const int nvirt = nbasis - nocc;

	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {
			Eigen::MatrixXd K_mat(nvirt, nvirt);
			K_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd dij_mat(nvirt, nvirt);
			dij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					dij_mat(a - nocc, b - nocc) = dij(i, a, j, b);
					K_mat(a - nocc, b - nocc) = orb(i, a, j, b);
				}
			}

			Eigen::MatrixXd K_mat_PNO = dij_mat.transpose() * K_mat * dij_mat;

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					orb_PNO(i, a, j, b) = K_mat_PNO(a - nocc, b - nocc);
				}
			}

		}
	}

	return orb_PNO;
}

TensorRank6 virtual_orbital_transformation_to_PNO(const TensorRank4 &orb,
		const TensorRank4 &dij, const int nbasis, const int nocc) {

	const int nvirt = nbasis - nocc;
	TensorRank6 orb_virt_PNO(nbasis, nbasis, nbasis, nbasis, nbasis, nbasis);

	TensorRank4 orb_v_PNO_1(nbasis, nbasis, nbasis, nbasis);
	TensorRank4 orb_v_PNO_2(nbasis, nbasis, nbasis, nbasis);
	TensorRank4 orb_v_PNO_3(nbasis, nbasis, nbasis, nbasis);
	TensorRank4 orb_v_PNO_4(nbasis, nbasis, nbasis, nbasis);

	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {

			Eigen::MatrixXd dij_mat(nvirt, nvirt);
			dij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					dij_mat(a - nocc, b - nocc) = dij(i, a, j, b);
				}
			}

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {

					for (size_t c = nocc; c < nbasis; c++) {
						for (size_t dt = nocc; dt < nbasis; dt++) {
							double t1 = 0.0;
							for (size_t d = nocc; d < nbasis; d++) {
								t1 += dij_mat(d - nocc, dt - nocc)
										* orb(a, c, b, d);
							}
							orb_v_PNO_1(a, c, b, dt) = t1;
						}
					}
				}
			}

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {

					for (size_t dt = nocc; dt < nbasis; dt++) {
						for (size_t ct = nocc; ct < nbasis; ct++) {
							double t2 = 0;
							for (size_t c = nocc; c < nbasis; c++) {
								t2 += dij_mat(c - nocc, ct - nocc)
										* orb_v_PNO_1(a, c, b, dt);
							}
							orb_v_PNO_2(a, ct, b, dt) = t2;
						}
					}
				}
			}

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t dt = nocc; dt < nbasis; dt++) {

					for (size_t ct = nocc; ct < nbasis; ct++) {
						for (size_t bt = nocc; bt < nbasis; bt++) {
							double t3 = 0;
							for (size_t b = nocc; b < nbasis; b++) {
								t3 += dij_mat(b - nocc, bt - nocc)
										* orb_v_PNO_2(a, ct, b, dt);
							}
							orb_v_PNO_3(a, ct, bt, dt) = t3;
						}
					}
				}
			}

			for (size_t dt = nocc; dt < nbasis; dt++) {
				for (size_t ct = nocc; ct < nbasis; ct++) {

					for (size_t bt = nocc; bt < nbasis; bt++) {
						for (size_t at = nocc; at < nbasis; at++) {
							double t4 = 0;
							for (size_t a = nocc; a < nbasis; a++) {
								t4 += dij_mat(a - nocc, at - nocc)
										* orb_v_PNO_3(a, ct, bt, dt);
							}
							orb_v_PNO_4(at, ct, bt, dt) = t4;
						}
					}
				}
			}

			for (size_t at = nocc; at < nbasis; at++) {
				for (size_t bt = nocc; bt < nbasis; bt++) {
					for (size_t ct = nocc; ct < nbasis; ct++) {
						for (size_t dt = nocc; dt < nbasis; dt++) {
							orb_virt_PNO(i, j, at, ct, bt, dt) = orb_v_PNO_4(at,
									ct, bt, dt);
						}
					}
				}
			}

		}
	}

	return orb_virt_PNO;
}

TensorRank6 semi_virtual_orbital_transformation_to_PNO(const TensorRank4 &orb,
		const TensorRank4 &dij, const int nbasis, const int nocc) {

	const int nvirt = nbasis - nocc;
	TensorRank6 orb_s_virt_PNO(nbasis, nbasis, nbasis, nbasis, nbasis, nbasis);

	TensorRank4 orb_v_PNO_1(nbasis, nbasis, nbasis, nbasis);
	TensorRank4 orb_v_PNO_2(nbasis, nbasis, nbasis, nbasis);
	TensorRank4 orb_v_PNO_3(nbasis, nbasis, nbasis, nbasis);

	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {

			Eigen::MatrixXd dij_mat(nvirt, nvirt);
			dij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					dij_mat(a - nocc, b - nocc) = dij(i, a, j, b);
				}
			}

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {

					for (size_t ct = nocc; ct < nbasis; ct++) {
						double t1 = 0.0;
						for (size_t c = nocc; c < nbasis; c++) {
							t1 += dij_mat(c - nocc, ct - nocc)
									* orb(i, a, c, b);
						}
						orb_v_PNO_1(i, a, ct, b) = t1;
					}
				}
			}

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t ct = nocc; ct < nbasis; ct++) {

					for (size_t bt = nocc; bt < nbasis; bt++) {
						double t2 = 0.0;
						for (size_t b = nocc; b < nbasis; b++) {
							t2 += dij_mat(b - nocc, bt - nocc)
									* orb_v_PNO_1(i, a, ct, b);
						}
						orb_v_PNO_2(i, a, ct, bt) = t2;
					}
				}
			}

			for (size_t ct = nocc; ct < nbasis; ct++) {
				for (size_t bt = nocc; bt < nbasis; bt++) {

					for (size_t at = nocc; at < nbasis; at++) {
						double t3 = 0.0;
						for (size_t a = nocc; a < nbasis; a++) {
							t3 += dij_mat(a - nocc, at - nocc)
									* orb_v_PNO_2(i, a, ct, bt);
						}

						orb_v_PNO_3(i, at, ct, bt) = t3;
					}
				}
			}

			for (size_t at = nocc; at < nbasis; at++) {
				for (size_t bt = nocc; bt < nbasis; bt++) {
					for (size_t ct = nocc; ct < nbasis; ct++) {

						orb_s_virt_PNO(i, j, i, at, ct, bt) = orb_v_PNO_3(i, at,
								ct, bt);
					}
				}
			}

		}
	}

	return orb_s_virt_PNO;
}

Eigen::MatrixXd transformation_of_singles_amplitudes_into_PNO(Eigen::MatrixXd t,
		const TensorRank4 dij, const int nocc, const int nbasis) {

	const int nvirt = nbasis - nocc;
	Eigen::MatrixXd t_PNO(nocc, nvirt);

	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {

			Eigen::MatrixXd dij_mat(nvirt, nvirt);
			dij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					dij_mat(a - nocc, b - nocc) = dij(i, a, j, b);
				}
			}

			double num = 0.0;
			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t at = nocc; at < nbasis; at++) {
					num += dij_mat(at - nocc, a - nocc) * t(i, at - nocc);
				}
				t_PNO(i, a - nocc) = num;
			}

		}
	}

	return t_PNO;
}

TensorRank4 get_residual_LPNO_CEPA_doubles(std::string method,
		const TensorRank4 &orb, const TensorRank6 &orb_virt_PNO,
		const TensorRank6 &orb_s_virt_PNO, const TensorRank4 Tij_PNO,
		Eigen::MatrixXd &t_PNO, const Eigen::MatrixXd &Ft, const int nocc,
		const int nbasis, const TensorRank4 dij) {

	TensorRank4 Res_doubles(nbasis, nbasis, nbasis, nbasis);
	TensorRank4 orb_at_PNO(nbasis, nbasis, nbasis, nbasis);
	TensorRank4 orb_bt_PNO(nbasis, nbasis, nbasis, nbasis);

	const int nvirt = nbasis - nocc;

	Eigen::MatrixXd F_virt(nvirt, nvirt);

	for (size_t a = nocc; a < nbasis; a++) {
		for (size_t b = nocc; b < nbasis; b++) {
			F_virt(a - nocc, b - nocc) = Ft(a, b);
		}
	}

	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {

			Eigen::MatrixXd R_mat(nvirt, nvirt);
			R_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd dij_mat(nvirt, nvirt);
			dij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::MatrixXd dijt_mat(nvirt, nvirt);
			dijt_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::MatrixXd K_mat(nvirt, nvirt);
			K_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::MatrixXd Tij_mat_PNO(nvirt, nvirt);
			Tij_mat_PNO = Eigen::MatrixXd::Zero(nvirt, nvirt);

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					dij_mat(a - nocc, b - nocc) = dij(i, a, j, b);
					dijt_mat(a - nocc, b - nocc) = dij(i, b, j, a);
					K_mat(a - nocc, b - nocc) = orb(i, a, j, b);
					Tij_mat_PNO(a - nocc, b - nocc) = Tij_PNO(i, a, j, b);
				}
			}

			Eigen::MatrixXd term_1 = dij_mat.transpose() * K_mat * dij_mat;

			Eigen::MatrixXd term_2(nvirt, nvirt);
			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {

					double summ_virt = 0.0;
					for (size_t c = nocc; c < nbasis; c++) {
						for (size_t d = nocc; d < nbasis; d++) {

							summ_virt += orb_virt_PNO(i, j, a, c, b, d)
									* Tij_PNO(i, c, j, d);
						}
					}

					term_2(a - nocc, b - nocc) = summ_virt;
				}
			}

			Eigen::MatrixXd F_virt_PNO = dij_mat.transpose() * F_virt * dij_mat;
			Eigen::MatrixXd term_3 = F_virt_PNO * Tij_mat_PNO
					+ Tij_mat_PNO * F_virt_PNO;

			Eigen::MatrixXd term_4(nvirt, nvirt);
			term_4 = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd term_5(nvirt, nvirt);
			term_5 = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd term_6(nvirt, nvirt);
			term_6 = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd term_7(nvirt, nvirt);
			term_7 = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd term_8(nvirt, nvirt);
			term_8 = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd term_9(nvirt, nvirt);
			term_9 = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd term_10(nvirt, nvirt);
			term_10 = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd term_11(nvirt, nvirt);
			term_11 = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd Kia(nvirt, nvirt);
			Kia = Eigen::MatrixXd::Zero(nvirt, nvirt);

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {

					double summ1 = 0.0;
					for (size_t k = 0; k < nocc; k++) {

						Eigen::MatrixXd dik_mat(nvirt, nvirt);
						dik_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
						Eigen::MatrixXd dikt_mat(nvirt, nvirt);
						dikt_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd dkj_mat(nvirt, nvirt);
						dkj_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
						Eigen::MatrixXd dkjt_mat(nvirt, nvirt);
						dkjt_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd Tik_mat_PNO(nvirt, nvirt);
						Tik_mat_PNO = Eigen::MatrixXd::Zero(nvirt, nvirt);
						Eigen::MatrixXd Tkj_mat_PNO(nvirt, nvirt);
						Tkj_mat_PNO = Eigen::MatrixXd::Zero(nvirt, nvirt);

						for (size_t c = nocc; c < nbasis; c++) {
							for (size_t d = nocc; d < nbasis; d++) {
								dik_mat(c - nocc, d - nocc) = dij(i, c, k, d);
								dikt_mat(c - nocc, d - nocc) = dij(i, d, k, c);
								dkj_mat(c - nocc, d - nocc) = dij(k, c, j, d);
								dkjt_mat(c - nocc, d - nocc) = dij(k, d, j, c);

								Tik_mat_PNO(c - nocc, d - nocc) = Tij_PNO(i, c,
										k, d);
								Tkj_mat_PNO(c - nocc, d - nocc) = Tij_PNO(k, c,
										j, d);
							}
						}
						Eigen::MatrixXd Sijik = dij_mat.transpose() * dik_mat;
						Eigen::MatrixXd Sijikt = dij_mat.transpose() * dikt_mat;

						Eigen::MatrixXd Sijkj = dij_mat.transpose() * dkj_mat;
						Eigen::MatrixXd Sijkjt = dij_mat.transpose() * dkjt_mat;

						//T^(ik)_tilda = S^(ij,ik)*T^(ik)_PNO*S^(ij,ikt)

						Eigen::MatrixXd Tik_mat_tilda = Sijik * Tik_mat_PNO
								* Sijik.transpose();
						Eigen::MatrixXd Tkj_mat_tilda = Sijkj * Tkj_mat_PNO
								* Sijkj.transpose();

						summ1 += Ft(j, k) * Tik_mat_tilda(a - nocc, b - nocc)
								+ Tkj_mat_tilda(a - nocc, b - nocc) * Ft(i, k);
					}

					term_4(a - nocc, b - nocc) = summ1;

					double summ2 = 0.0;
					for (size_t k = 0; k < nocc; k++) {
						for (size_t l = 0; l < nocc; l++) {

							Eigen::MatrixXd dkl_mat(nvirt, nvirt);
							dkl_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
							Eigen::MatrixXd dklt_mat(nvirt, nvirt);
							dklt_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

							Eigen::MatrixXd Tkl_mat_PNO(nvirt, nvirt);
							Tkl_mat_PNO = Eigen::MatrixXd::Zero(nvirt, nvirt);

							for (size_t c = nocc; c < nbasis; c++) {
								for (size_t d = nocc; d < nbasis; d++) {
									dkl_mat(c - nocc, d - nocc) = dij(k, c, l,
											d);
									dklt_mat(c - nocc, d - nocc) = dij(k, d, l,
											c);

									Tkl_mat_PNO(c - nocc, d - nocc) = Tij_PNO(k,
											c, l, d);
								}
							}
							Eigen::MatrixXd Sijkl = dij_mat.transpose()
									* dkl_mat;
							Eigen::MatrixXd Sijklt = dij_mat.transpose()
									* dklt_mat;
							Eigen::MatrixXd Tkl_mat_tilda = Sijkl * Tkl_mat_PNO
									* Sijkl.transpose();
							summ2 += orb(i, k, j, l)
									* Tkl_mat_tilda(a - nocc, b - nocc);
						}
					}

					term_5(a - nocc, b - nocc) = summ2;

					double summ3 = 0.0;
					for (size_t k = 0; k < nocc; k++) {

						Eigen::MatrixXd dik_mat(nvirt, nvirt);
						dik_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd dkit_mat(nvirt, nvirt);
						dkit_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd Tik_mat_PNO(nvirt, nvirt);
						Tik_mat_PNO = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd Tikt_mat_PNO(nvirt, nvirt);
						Tikt_mat_PNO = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd Kkj_mat(nvirt, nvirt);
						Kkj_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd Jkj_mat(nvirt, nvirt);
						Jkj_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd Kik_mat(nvirt, nvirt);
						Kik_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd Jik_mat(nvirt, nvirt);
						Jik_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd Tkj_mat_PNO(nvirt, nvirt);
						Tkj_mat_PNO = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd Tkjt_mat_PNO(nvirt, nvirt);
						Tkjt_mat_PNO = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd dkj_mat(nvirt, nvirt);
						dkj_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd djk_mat(nvirt, nvirt);
						djk_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

						for (size_t c = nocc; c < nbasis; c++) {
							for (size_t d = nocc; d < nbasis; d++) {
								dik_mat(c - nocc, d - nocc) = dij(i, c, k, d);
								dkit_mat(c - nocc, d - nocc) = dij(k, d, i, c);
								Tik_mat_PNO(c - nocc, d - nocc) = Tij_PNO(i, c,
										k, d);
								Tikt_mat_PNO(c - nocc, d - nocc) = Tij_PNO(i, d,
										k, c);
								Kkj_mat(c - nocc, d - nocc) = orb(k, c, j, d);
								Jkj_mat(c - nocc, d - nocc) = orb(k, j, c, d);
								Kik_mat(c - nocc, d - nocc) = orb(i, c, k, d);
								Jik_mat(c - nocc, d - nocc) = orb(i, k, c, d);
								Tkj_mat_PNO(c - nocc, d - nocc) = Tij_PNO(k, c,
										j, d);
								Tkjt_mat_PNO(c - nocc, d - nocc) = Tij_PNO(k, d,
										j, c);
								dkj_mat(c - nocc, d - nocc) = dij(k, c, j, d);
								djk_mat(c - nocc, d - nocc) = dij(j, c, k, d);
							}
						}

						Eigen::MatrixXd Sijik = dij_mat.transpose() * dik_mat;
						Eigen::MatrixXd Sijkj = dij_mat.transpose() * dkj_mat;

						Eigen::MatrixXd intermediate = Sijik
								* (2 * Tik_mat_PNO - Tikt_mat_PNO) * dkit_mat
								* (Kkj_mat - 0.5 * Jkj_mat) * dij_mat
								+ dijt_mat * (Kik_mat - 0.5 * Jik_mat) * djk_mat
										* (2 * Tkj_mat_PNO - Tkjt_mat_PNO)
										* Sijkj.transpose();

						summ3 += intermediate(a - nocc, b - nocc);
					}

					term_6(a - nocc, b - nocc) = summ3;

					double summ4 = 0.0;
					for (size_t k = 0; k < nocc; k++) {

						Eigen::MatrixXd dik_mat(nvirt, nvirt);
						dik_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd dkit_mat(nvirt, nvirt);
						dkit_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd dijt_mat(nvirt, nvirt);
						dijt_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd dki_mat(nvirt, nvirt);
						dki_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd dkj_mat(nvirt, nvirt);
						dkj_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd dkjt_mat(nvirt, nvirt);
						dkjt_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd Tikt_mat_PNO(nvirt, nvirt);
						Tikt_mat_PNO = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd Tkjt_mat_PNO(nvirt, nvirt);
						Tkjt_mat_PNO = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd Tik_mat_PNO(nvirt, nvirt);
						Tik_mat_PNO = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd Tkj_mat_PNO(nvirt, nvirt);
						Tkj_mat_PNO = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd Jjkt_mat(nvirt, nvirt);
						Jjkt_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd Jik_mat(nvirt, nvirt);
						Jik_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd Jjk_mat(nvirt, nvirt);
						Jjk_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

						Eigen::MatrixXd Jikt_mat(nvirt, nvirt);
						Jikt_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

						for (size_t c = nocc; c < nbasis; c++) {
							for (size_t d = nocc; d < nbasis; d++) {
								dik_mat(c - nocc, d - nocc) = dij(i, c, k, d);
								dkit_mat(c - nocc, d - nocc) = dij(k, d, i, c);
								dijt_mat(c - nocc, d - nocc) = dij(i, d, j, c);
								dki_mat(c - nocc, d - nocc) = dij(k, c, i, d);
								dkj_mat(c - nocc, d - nocc) = dij(k, c, j, d);
								dkjt_mat(c - nocc, d - nocc) = dij(k, d, j, c);
								Tikt_mat_PNO(c - nocc, d - nocc) = Tij_PNO(i, d,
										k, c);
								Tkjt_mat_PNO(c - nocc, d - nocc) = Tij_PNO(k, d,
										j, c);
								Tik_mat_PNO(c - nocc, d - nocc) = Tij_PNO(i, c,
										k, d);
								Tkj_mat_PNO(c - nocc, d - nocc) = Tij_PNO(k, c,
										j, d);
								Jjkt_mat(c - nocc, d - nocc) = orb(j, k, d, c);
								Jik_mat(c - nocc, d - nocc) = orb(i, k, c, d);
								Jjk_mat(c - nocc, d - nocc) = orb(j, k, c, d);
								Jikt_mat(c - nocc, d - nocc) = orb(i, k, d, c);
							}
						}

						Eigen::MatrixXd Sijik = dij_mat.transpose() * dik_mat;
						Eigen::MatrixXd Skjij = dkj_mat.transpose() * dij_mat;
						Eigen::MatrixXd Sikij = dik_mat.transpose() * dij_mat;
						Eigen::MatrixXd Sijkj = dij_mat.transpose() * dkj_mat;

						Eigen::MatrixXd intermediate = 0.5 * Sijik
								* Tikt_mat_PNO * dkit_mat * Jjkt_mat * dij_mat
								+ 0.5 * dijt_mat * Jik_mat * dkj_mat
										* Tkjt_mat_PNO * Skjij
								+ dijt_mat * Jjk_mat * dki_mat * Tik_mat_PNO
										* Sikij
								+ Sijkj * Tkj_mat_PNO * dkjt_mat * Jikt_mat
										* dij_mat;

						summ4 += intermediate(a - nocc, b - nocc);
					}

					term_7(a - nocc, b - nocc) = summ4;

					double summ5 = 0.0;
					//wrong sign
					for (size_t k = 0; k < nocc; k++) {
						//or other way around for t
						double t1 = 0.0;
						for (size_t at = nocc; at < nbasis; at++) {
							t1 += orb(j, k, i, at)
									* dij_mat(at - nocc, a - nocc);
						}
						orb_at_PNO(j, k, i, a) = t1;

						double t2 = 0.0;
						for (size_t bt = nocc; bt < nbasis; bt++) {
							t2 += orb(i, k, j, bt)
									* dij_mat(bt - nocc, b - nocc);
						}
						orb_bt_PNO(i, k, j, b) = t2;

						summ5 += orb_bt_PNO(i, k, j, b) * t_PNO(k, a - nocc)
								+ orb_at_PNO(j, k, i, a) * t_PNO(k, b - nocc);
					}
					term_9(a - nocc, b - nocc) = summ5;

					double summ6 = 0.0;
					//wrong sign
					for (size_t c = nocc; c < nbasis; c++) {

						double t1 = 0.0;
						double t2 = 0.0;
						for (size_t ct = nocc; ct < nbasis; ct++) {
							for (size_t bt = nocc; bt < nbasis; bt++) {
								for (size_t at = nocc; at < nbasis; at++) {
									t1 += dij_mat(at - nocc, a - nocc)
											* dij_mat(bt - nocc, b - nocc)
											* dij_mat(ct - nocc, c - nocc)
											* orb(j, bt, at, ct);
									t2 += dij_mat(at - nocc, a - nocc)
											* dij_mat(bt - nocc, b - nocc)
											* dij_mat(ct - nocc, c - nocc)
											* orb(i, at, ct, bt);
								}
							}
						}
						summ6 += t1 * t_PNO(i, c - nocc)
								+ t2 * t_PNO(j, c - nocc);

					}
					term_10(a - nocc, b - nocc) = summ6;

					/*
					 //CEPA/0 method
					 if (method == "LPNO-CEPA/0") {
					 term_11(a - nocc, b - nocc) = 0.0;
					 }
					 */
				}

			}

			R_mat = term_1 + term_2 + term_3 - term_4 + term_5 + term_6 - term_7
					- term_9 + term_10;

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					Res_doubles(i, a, j, b) = R_mat(a - nocc, b - nocc);
				}
			}

		}
	}

	return Res_doubles;
}

Eigen::MatrixXd get_residual_LPNO_CEPA_singles(std::string method,
		const TensorRank4 &orb, const TensorRank4 & Tij,
		const TensorRank4 &Tij_PNO, Eigen::MatrixXd &t,
		const Eigen::MatrixXd &Ft, const int nocc, const int nbasis,
		const TensorRank4 dij) {

	Eigen::MatrixXd Res_singles(nbasis, nbasis);

	const int nvirt = nbasis - nocc;

	Eigen::MatrixXd F_virt(nvirt, nvirt);

	for (size_t a = nocc; a < nbasis; a++) {
		for (size_t b = nocc; b < nbasis; b++) {
			F_virt(a - nocc, b - nocc) = Ft(a, b);
		}
	}

	Eigen::MatrixXd K_mat(nocc, nvirt);
	Eigen::MatrixXd Tij_mat(nvirt, nvirt);

	Eigen::MatrixXd term_1(nocc, nvirt);
	term_1 = Eigen::MatrixXd::Zero(nocc, nvirt);

	Eigen::MatrixXd term_2(nocc, nvirt);
	term_2 = Eigen::MatrixXd::Zero(nocc, nvirt);

	Eigen::MatrixXd term_3(nocc, nvirt);
	term_3 = Eigen::MatrixXd::Zero(nocc, nvirt);

	Eigen::MatrixXd term_4(nocc, nvirt);
	term_4 = Eigen::MatrixXd::Zero(nocc, nvirt);

	Eigen::MatrixXd term_5(nocc, nvirt);
	term_5 = Eigen::MatrixXd::Zero(nocc, nvirt);

	Eigen::MatrixXd term_6(nocc, nvirt);
	term_6 = Eigen::MatrixXd::Zero(nocc, nvirt);

	Eigen::MatrixXd term_7(nocc, nvirt);
	term_7 = Eigen::MatrixXd::Zero(nocc, nvirt);

	Eigen::VectorXd Fv(nvirt);
	Fv = Eigen::VectorXd::Zero(nvirt);

	Eigen::VectorXd t_vec(nvirt);
	t_vec = Eigen::VectorXd::Zero(nvirt);

	for (size_t i = 0; i < nocc; i++) {

		for (size_t a = nocc; a < nbasis; a++) {

			term_1(i, a - nocc) = Ft(i, a);
		}

		//or other way around
		//Eigen::MatrixXd term_2 =  t.transpose()*Fv;

		for (size_t a = nocc; a < nbasis; a++) {
			double summ0 = 0.0;

			for (size_t e = nocc; e < nbasis; e++) {
				summ0 += Ft(a, e) * t(i, e - nocc);
			}
			term_2(i, a - nocc) = summ0;
		}

		for (size_t a = nocc; a < nbasis; a++) {

			double summ1 = 0.0;

			for (size_t j = 0; j < nocc; j++) {
				summ1 += Ft(i, j) * t(j, a - nocc);
			}

			term_3(i, a - nocc) = summ1;
		}

		for (size_t a = nocc; a < nbasis; a++) {

			double summ2 = 0.0;
			//wrong sign
			for (size_t j = 0; j < nocc; j++) {
				for (size_t k = 0; k < nocc; k++) {
					for (size_t b = nocc; b < nbasis; b++) {
						summ2 += (2 * orb(i, j, k, b) - orb(i, k, j, b))
								* Tij(k, b, j, a);
					}
				}
			}

			term_4(i, a - nocc) = summ2;
		}

		for (size_t a = nocc; a < nbasis; a++) {

			double summ3 = 0.0;
			for (size_t j = 0; j < nocc; j++) {
				for (size_t e = nocc; e < nbasis; e++) {

					summ3 += (2 * orb(i, a, j, e) - orb(i, j, a, e))
							* t(j, e - nocc);
				}
			}

			term_5(i, a - nocc) = summ3;
		}

		for (size_t a = nocc; a < nbasis; a++) {

			double summ4 = 0.0;
			for (size_t j = 0; j < nocc; j++) {
				for (size_t e = nocc; e < nbasis; e++) {

					summ4 += Ft(j, e) * (2 * Tij(i, a, j, e) - Tij(i, e, j, a));
				}
			}

			term_6(i, a - nocc) = summ4;
		}

		for (size_t a = nocc; a < nbasis; a++) {

			double summ4 = 0.0;
			for (size_t j = 0; j < nocc; j++) {

				for (size_t bt = nocc; bt < nbasis; bt++) {
					for (size_t ct = nocc; ct < nbasis; ct++) {

						double t1 = 0.0;
						double t2 = 0.0;
						for (size_t c = nocc; c < nbasis; c++) {
							for (size_t b = nocc; b < nbasis; b++) {
								t1 += dij(i, b, j, bt) * dij(i, c, j, ct)
										* orb(j, b, a, c);
								t2 += dij(i, b, j, bt) * dij(i, c, j, ct)
										* orb(j, c, a, b);
							}
						}

						summ4 += (2 * t1 - t2) * Tij_PNO(i, ct, j, bt);

					}
				}
			}
			term_7(i, a - nocc) = summ4;

		}

	}

	Res_singles = term_1 + term_2 - term_3 - term_4 + term_5 + term_6 + term_7;

	return Res_singles;
}

TensorRank4 transformation_of_doubles_amplitudes_from_PNO_into_MO(
		TensorRank4 Tij_PNO, const TensorRank4 dij, const int nocc,
		const int nbasis) {

	TensorRank4 Tij(nbasis, nbasis, nbasis, nbasis);
	const int nvirt = nbasis - nocc;

	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {

			Eigen::MatrixXd dij_mat(nvirt, nvirt);
			dij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::MatrixXd dijt_mat(nvirt, nvirt);
			dijt_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::MatrixXd Tij_mat_PNO(nvirt, nvirt);
			Tij_mat_PNO = Eigen::MatrixXd::Zero(nvirt, nvirt);

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					dij_mat(a - nocc, b - nocc) = dij(i, a, j, b);
					dijt_mat(a - nocc, b - nocc) = dij(i, b, j, a);
					Tij_mat_PNO(a - nocc, b - nocc) = Tij_PNO(i, a, j, b);
				}
			}

			Eigen::MatrixXd Tij_mat = dij_mat * Tij_mat_PNO
					* dij_mat.transpose();

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					Tij(i, a, j, b) = Tij_mat(a - nocc, b - nocc);
				}
			}

		}
	}

	return Tij;
}

void LPNO_CEPA_linear_solver(const TensorRank4 &orb, const Eigen::MatrixXd &F,
		const Eigen::MatrixXd &C, TensorRank4 Tij, const int nocc,
		const int nbasis, const TensorRank4 &dij) {

	Eigen::MatrixXd Ft = C.transpose() * F * C;
//std::cout << Ft << std::endl;

	const int nvirt = nbasis - nocc;

	Eigen::VectorXd e_s(nocc);
	Eigen::MatrixXd e_d(nocc, nocc);
	e_d = Eigen::MatrixXd::Zero(nocc, nocc);

//transformation of orbitals into PNO basis
	TensorRank4 orb_PNO = orbital_transformation_to_PNO(orb, dij, nbasis, nocc);

//transformation of virtual orbitals (a c | b d) into PNO basis (at ct | bt dt)
	TensorRank6 orb_virt_PNO = virtual_orbital_transformation_to_PNO(orb, dij,
			nbasis, nocc);

//transformation of semi virtual orbitals (i a | c b) into PNO basis (i at | ct bt)
	TensorRank6 orb_s_virt_PNO = semi_virtual_orbital_transformation_to_PNO(orb,
			dij, nbasis, nocc);

	TensorRank4 Tij_PNO = transformation_of_doubles_amplitudes_from_MO_into_PNO(
			Tij, dij, nocc, nbasis);

	Eigen::MatrixXd t(nocc, nbasis - nocc);
//t = Eigen::MatrixXd::Zero(nocc, nbasis - nocc);

	for (size_t i = 0; i < nocc; i++) {
		for (size_t a = nocc; a < nbasis; a++) {
			t(i, a - nocc) = 1.0;
		}
	}

	Eigen::MatrixXd t_PNO = transformation_of_singles_amplitudes_into_PNO(t,
			dij, nocc, nbasis);

	double E_LPNO_CEPA = 0.0;
	int count = 0;

	std::string method;
	method = "LPNO-CEPA/0";

	std::cout << "\n" << std::endl;

	while (true) {

		TensorRank4 LPNO_Res_doubles = get_residual_LPNO_CEPA_doubles(method,
				orb, orb_virt_PNO, orb_s_virt_PNO, Tij_PNO, t_PNO, Ft, nocc,
				nbasis, dij);

		Eigen::MatrixXd Res_singles = get_residual_CEPA_singles(method, orb,
				Tij, t, Ft, nocc, nbasis, count, E_LPNO_CEPA, e_d);

		/*
		 Eigen::MatrixXd Res_singles = get_residual_LPNO_CEPA_singles(method,
		 orb, Tij, Tij_PNO, t, Ft, nocc, nbasis, dij);
		 */
		TensorRank4 dT_PNO = get_increment_of_amplitude_PNO(LPNO_Res_doubles,
				dij, Ft, C, nocc, nbasis);

		Eigen::MatrixXd dt = get_amplitude_increment_singles(Res_singles, Ft,
				nocc, nbasis);

		Tij_PNO = get_new_amplitudes_doubles(Tij_PNO, dT_PNO, nocc, nbasis);

		Tij = transformation_of_doubles_amplitudes_from_PNO_into_MO(Tij_PNO,
				dij, nocc, nbasis);

		t = get_new_amplitudes_singles(t, dt, nocc, nbasis);

		t_PNO = transformation_of_singles_amplitudes_into_PNO(t, dij, nocc,
				nbasis);

		E_LPNO_CEPA = get_energies_CEPA(orb_PNO, Tij_PNO, nocc, nbasis);

		double max_d = max_abs_Res_doubles(LPNO_Res_doubles, nocc, nbasis);
		double max_s = max_abs_Res_singles(Res_singles, nocc, nbasis);

		std::cout << "Energy: " << E_LPNO_CEPA << "   Double Residual: "
				<< max_d << "   Single Residual: " << max_s << std::endl;

		count++;

		//std::cout << count << std::endl;
		if (max_d < 1e-6 && max_s < 1e-6)
			//if (max_d < 1e-10)
			break;

	}

	std::cout << method << "\t" << "energy" << std::endl;
	std::cout << E_LPNO_CEPA << std::endl;
}

TensorRank4 get_residual_CCSD_doubles(const TensorRank4 &orb,
		const TensorRank4 Tij, Eigen::MatrixXd &t, const Eigen::MatrixXd &Ft,
		const int nocc, const int nbasis) {

	TensorRank4 Res_doubles(nbasis, nbasis, nbasis, nbasis);

	const int nvirt = nbasis - nocc;

	Eigen::MatrixXd F_virt(nvirt, nvirt);

	for (size_t a = nocc; a < nbasis; a++) {
		for (size_t b = nocc; b < nbasis; b++) {
			F_virt(a - nocc, b - nocc) = Ft(a, b);
		}
	}

	for (size_t i = 0; i < nocc; i++) {
		for (size_t j = 0; j < nocc; j++) {

			Eigen::MatrixXd R_mat(nvirt, nvirt);
			R_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd K_mat(nvirt, nvirt);
			K_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::MatrixXd K_virt(nvirt, nvirt);
			K_virt = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::MatrixXd Tij_mat(nvirt, nvirt);
			Tij_mat = Eigen::MatrixXd::Zero(nvirt, nvirt);
			Eigen::MatrixXd Tij_virt(nvirt, nvirt);
			Tij_virt = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd tau(nvirt, nvirt);
			tau = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd term_2(nvirt, nvirt);
			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {

					double summ_virt = 0.0;
					for (size_t c = nocc; c < nbasis; c++) {
						for (size_t d = nocc; d < nbasis; d++) {
							K_virt(c - nocc, d - nocc) = orb(a, c, b, d);

							double summ0 = 0.0;
							for (size_t k = 0; k < nocc; k++) {
								summ0 += Tij(k, d, a, c) * t(k, b - nocc)
										+ Tij(k, c, b, d) * t(k, a - nocc);
							}
							summ_virt += (K_virt(c - nocc, d - nocc) + summ0)
									* (Tij(i, c, j, d)
											+ t(i, c - nocc) * t(j, d - nocc));
						}
					}

					term_2(a - nocc, b - nocc) = summ_virt;
					K_mat(a - nocc, b - nocc) = orb(i, a, j, b);
					Tij_mat(a - nocc, b - nocc) = Tij(i, a, j, b);

					tau(a - nocc, b - nocc) = Tij(i, a, j, b)
							+ t(i, a - nocc) * t(j, b - nocc);
				}
			}

			Eigen::MatrixXd term_1 = K_mat;

			Eigen::MatrixXd term_3 = F_virt * Tij_mat + Tij_mat * F_virt;

			Eigen::MatrixXd term_4(nvirt, nvirt);
			term_4 = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd term_5(nvirt, nvirt);
			term_5 = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd term_6(nvirt, nvirt);
			term_6 = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd term_7(nvirt, nvirt);
			term_7 = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd term_8(nvirt, nvirt);
			term_8 = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd term_9(nvirt, nvirt);
			term_9 = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd term_10(nvirt, nvirt);
			term_10 = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd term_11(nvirt, nvirt);
			term_11 = Eigen::MatrixXd::Zero(nvirt, nvirt);

			Eigen::MatrixXd Kia(nvirt, nvirt);
			Kia = Eigen::MatrixXd::Zero(nvirt, nvirt);

			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {

					double summ1 = 0.0;
					for (size_t k = 0; k < nocc; k++) {
						summ1 += Ft(j, k) * Tij(i, a, k, b)
								+ Ft(i, k) * Tij(k, a, j, b);
					}

					term_4(a - nocc, b - nocc) = summ1;

					double summ2 = 0.0;
					for (size_t k = 0; k < nocc; k++) {
						for (size_t l = 0; l < nocc; l++) {
							//for (size_t k = nocc; k < nbasis; k++) {
							//for (size_t l = nocc; l < nbasis; l++) {
							summ2 += orb(i, k, j, l) * Tij(k, a, l, b);
						}
					}
					term_5(a - nocc, b - nocc) = summ2;

					double summ3 = 0.0;
					for (size_t k = 0; k < nocc; k++) {
						for (size_t e = nocc; e < nbasis; e++) {

							summ3 += (2 * Tij(i, a, k, e) - Tij(i, e, k, a))
									* (orb(k, e, j, b) - 0.5 * orb(k, j, e, b))
									+ (orb(i, a, k, e) - 0.5 * orb(i, k, a, e))
											* (2 * Tij(k, e, j, b)
													- Tij(k, b, j, e));
						}
					}

					term_6(a - nocc, b - nocc) = summ3;

					double summ4 = 0.0;
					for (size_t k = 0; k < nocc; k++) {
						for (size_t e = nocc; e < nbasis; e++) {

							summ4 += 0.5 * Tij(i, e, k, a) * orb(j, k, b, e)
									+ 0.5 * orb(i, k, a, e) * Tij(k, b, j, e)
									+ orb(j, k, a, e) * Tij(i, e, k, b)
									+ Tij(k, a, j, e) * orb(i, k, b, e);
							//summ4 += 0.5 * Tij(i, e, k, a) * orb(j, k, b, e);
						}
					}

					term_7(a - nocc, b - nocc) = summ4;

					term_8(a - nocc, b - nocc) = t(i, a - nocc) * Ft(j, b)
							+ t(j, b - nocc) * Ft(i, a);

					double summ5 = 0.0;
					//wrong sign
					for (size_t k = 0; k < nocc; k++) {
						//or other way around for t
						summ5 += orb(i, k, j, b) * t(k, a - nocc)
								+ orb(j, k, i, a) * t(k, b - nocc);
					}
					term_9(a - nocc, b - nocc) = summ5;

					double summ6 = 0.0;
					//wrong sign
					for (size_t e = nocc; e < nbasis; e++) {
						summ6 += orb(j, b, a, e) * t(i, e - nocc)
								+ orb(i, a, e, b) * t(j, e - nocc);

					}
					term_10(a - nocc, b - nocc) = summ6;

				}

			}

			R_mat = term_1 + term_2;

			//R_mat = term_1 + term_2 + term_3 - term_4 + term_5 + term_6 - term_7
			//	- term_9 + term_10;
			//std::cout << "Res for pair i,j = " << i << "," << j << std::endl;
			//std::cout << term_10 << std::endl;
			for (size_t a = nocc; a < nbasis; a++) {
				for (size_t b = nocc; b < nbasis; b++) {
					Res_doubles(i, a, j, b) = R_mat(a - nocc, b - nocc);
				}
			}

		}

	}

	return Res_doubles;
}

void CCSD_linear_solver(const TensorRank4 &orb, const Eigen::MatrixXd &F,
		const Eigen::MatrixXd &C, TensorRank4 Tij, const int nocc,
		const int nbasis) {

	Eigen::MatrixXd Ft = C.transpose() * F * C;

	double nvirt = nbasis - nocc;

	Eigen::MatrixXd t(nocc, nbasis - nocc);
//t = Eigen::MatrixXd::Zero(nocc, nbasis - nocc);

	for (size_t i = 0; i < nocc; i++) {
		for (size_t a = nocc; a < nbasis; a++) {
			t(i, a - nocc) = 0.0;
		}
	}

	double E_CCSD = 0.0;
	int count = 0;

	std::cout << "\n" << std::endl;
	//while (true) {

	TensorRank4 Res_doubles = get_residual_CCSD_doubles(orb, Tij, t, Ft, nocc,
			nbasis);

	//Eigen::MatrixXd Res_singles = get_residual_CEPA_singles(method, orb,
	//	Tij, t, Ft, nocc, nbasis, count, E_CEPA, e_d);

	TensorRank4 dT = get_amplitude_increment_doubles(Res_doubles, Ft, nocc,
			nbasis);

	//Eigen::MatrixXd dt = get_amplitude_increment_singles(Res_singles, Ft,
	//	nocc, nbasis);

	Tij = get_new_amplitudes_doubles(Tij, dT, nocc, nbasis);

	//t = get_new_amplitudes_singles(t, dt, nocc, nbasis);

	//E_CCSD = get_energies_CEPA(orb, Tij, nocc, nbasis);

	//std::cout << E_CEPA << std::endl;

	double max_d = max_abs_Res_doubles(Res_doubles, nocc, nbasis);
	//double max_s = max_abs_Res_singles(Res_singles, nocc, nbasis);

	count++;

	//std::cout << "Correlation Energy: " << E_CEPA << "  Residual_double: " << max_d << std::endl;
	//if (max_d < 1e-8 && max_s < 1e-8)
	//	if (max_d < 1e-8)
	//break;

	//}

	//std::cout << "CCSD" << "\t" << "energy" << std::endl;
	//std::cout << E_CCSD << std::endl;
}

int main(int argc, char** argv) {

//timing of calculation
	clock_t t_0, t_n;
	t_0 = clock();

//
// construct a Molecule object
//
	Ref<Molecule> mol = new Molecule;

//H2
	/*
	 mol->add_atom(1, 0.0, 0.7, 1.0);
	 mol->add_atom(1, 0.0, -0.7, 1.0);
	 */

//He dimer at equilibrium bond distance
	/*
	 mol->add_atom(2, 0.0, 0.0, 0.0);
	 mol->add_atom(2, 0.0, 0.0, 5.6);
	 */

	//Ne dimer at equilibrium bond distance
	/*
	 mol->add_atom(10, 0.0, 0.0, 0.0);
	 mol->add_atom(10, 0.0, 0.0, 3.1);
	 */

	//ethylene molecule 90 degrees
	/*
	 mol->add_atom(1, -0.27110389,	1.23456186,	0.43510188);
	 mol->add_atom(6, -0.00035338,	0.00011905,	2.05562518);
	 mol->add_atom(6, 2.13647711,	-0.00015874,	3.38222804);
	 mol->add_atom(1, 2.40726920,	1.23368502,	5.00320109);
	 mol->add_atom(1, -1.57311952,	-1.23394014,	2.53181160);
	 mol->add_atom(1, 3.70911286,	-1.23426706,	2.90573360);
	 */

	//ethylene molecule 0 degrees
	/*
	 mol->add_atom(1, 0.00000000,	0.00000000,	0.00000000);
	 mol->add_atom(6, 0.00000000,	0.00000000,	2.05505448);
	 mol->add_atom(6, 2.13609350,	0.00000000,	3.38284598);
	 mol->add_atom(1, 2.13609350,	0.00000000,	5.43790046);
	 mol->add_atom(1, -1.84280422,	0.00011338,	2.96462880);
	 mol->add_atom(1, 3.97889772,	-0.00011338,	2.47327166);
	 */


	 //water
	 mol->add_atom(8, 0.0, 0.0, 0.0);
	 mol->add_atom(1, 0.0, 1.0, 1.0);
	 mol->add_atom(1, 0.0,-1.0, 1.0);


/*
	 //water molecule Harley
	 mol->add_atom(8, 0.00000000, 0.00000000, -0.12947689);
	 mol->add_atom(1, 0.00000000, -1.49418674, 1.02744610);
	 mol->add_atom(1, 0.00000000, 1.49418674, 1.02744610);
*/

	/*
//cyclohexatriene
	mol->add_atom(1, 0.00000000,	-2.68462241,	3.02634537);
	mol->add_atom(6,  0.00000000,	1.26643776,	1.53978475);
	mol->add_atom(6, 0.00000000,	-1.26643776,	1.53978475);
	mol->add_atom(6, 0.00000000,	1.48591244,	-1.32223570);
	mol->add_atom(6, 0.00000000,	-1.48591244,	-1.32223570);
	mol->add_atom(1, 1.68206223,	-2.35573070,	-2.16581700);
	mol->add_atom(1, -1.68206223,	-2.35573070,	-2.16581700);
	mol->add_atom(1, 0.00000000,	2.68462241,	3.02634537);
	mol->add_atom(1, -1.68206223,	2.35573070,	-2.16581700);
	mol->add_atom(1, 1.68206223,	2.35573070,	-2.16581700);
*/
	/*		//water molecule from gaussian optimized HF/sto-3g
	 mol->add_atom(8,	0.00000000,	0.24032025,	0.00000000);
	 mol->add_atom(1,	1.43260704, -0.96128289,	0.00000000);
	 mol->add_atom(1,	-1.43260704, -0.96128289,	0.00000000);
	 */

	/*
	 //water molecule from mpqc mp2
	 mol->add_atom(8,	0.00000000,	0.00000000,	0.69919867);
	 mol->add_atom(1,	1.47398638,	0.00000000,	-0.34015070);
	 mol->add_atom(1,	-1.47398638,	0.00000000,	-0.34015070);
	 */
//naphtalene
/*
	 mol->add_atom(6,	4.57511954,	1.34692119,	-0.00009638);
	 mol->add_atom(6,	4.57520647,	-1.34681537,	0.00011338);
	 mol->add_atom(6,	2.36699913,	-2.63665549,	-0.00001134);
	 mol->add_atom(6,	0.00004157,	-1.32748536,	-0.00019653);
	 mol->add_atom(6,	-0.00009260,	1.32748536,	-0.00010016);
	 mol->add_atom(6,	2.36682339,	2.63672919,	-0.00023622);
	 mol->add_atom(1,	2.35116701,	4.68274134,	0.00033259);
	 mol->add_atom(6,	-2.36696701,	2.63666305,	0.00016819);
	 mol->add_atom(6,	-4.57520269,	1.34681348,	0.00009449);
	 mol->add_atom(6,	-4.57513466,	-1.34691174,	-0.00017952);
	 mol->add_atom(6,	-2.36679693,	-2.63674998,	0.00003213);
	 mol->add_atom(1,	-2.35116701,	-4.68276023,	0.00037795);
	 mol->add_atom(1,	-6.36212394,	-2.34302607,	0.00004157);
	 mol->add_atom(1,	-6.36225433,	2.34288056,	0.00022677);
	 mol->add_atom(1,	-2.35139189,	4.68265441,	0.00032314);
	 mol->add_atom(1,	2.35141078,	-4.68267331,	0.00031558);
	 mol->add_atom(1,	6.36226567,	-2.34284465,	0.00044787);
	 mol->add_atom(1,	6.36210126,	2.34306953,	0.00040251);
*/

	const bool use_symmetry = true;
	if (not use_symmetry) {
		Ref<PointGroup> c1_ptgrp = new PointGroup("C1");
		mol->set_point_group(c1_ptgrp);
	} else {
		mol->symmetrize(mol->highest_point_group(1e-4));
	}

	ExEnv::out0() << std::endl << indent << "constructed Molecule object:"
			<< std::endl;
	mol->print(ExEnv::out0());
	ExEnv::out0() << std::endl;

//
// construct a GaussianBasisSet object
//
	Ref<AssignedKeyVal> akv = new AssignedKeyVal;
	akv->assign("molecule", mol.pointer());
	akv->assign("name", "STO-3G");
	Ref<GaussianBasisSet> obs = new GaussianBasisSet(Ref<KeyVal>(akv));
// get rid of general constractions for simplicity
	if (obs->max_ncontraction() > 1) {
		Ref<GaussianBasisSet> split_basis = new SplitBasisSet(obs, obs->name());
		obs = split_basis;
	}

	const int nshell = obs->nshell();
	const int nbasis = obs->nbasis();

	std::cout << "nbasis" << nbasis << std::endl;
	ExEnv::out0() << std::endl << indent
			<< "constructed GaussianBasisSet object:" << std::endl;
	obs->print(ExEnv::out0());
	ExEnv::out0() << std::endl;

//
// construct an Integral object
// it will produce integral evaluator objects
//
	Ref<Integral> integral = Integral::initial_integral(argc, argv);
	if (integral.nonnull())
		Integral::set_default_integral(integral);
	integral = Integral::get_default_integral()->clone();
	integral->set_basis(obs);

//
// compute overlap integrals
//

// construct an OneBodyInt object that computes overlap integrals
	Eigen::MatrixXd S_mat(nbasis, nbasis);
	Ref<OneBodyInt> s_inteval = integral->overlap();
	const double* buffer = s_inteval->buffer();
// and compute overlap integrals
	get_overlap_ints(s_inteval, S_mat);

	s_inteval = 0;

//
// compute core Hamiltonian integrals
//

// construct an OneBodyInt object that computes overlap integrals
	Eigen::MatrixXd Hcore_mat(nbasis, nbasis);
	Ref<OneBodyInt> h_inteval = integral->hcore();
	buffer = h_inteval->buffer();
// and compute core Hamiltonian integrals
	get_core_hamiltonian_ints(h_inteval, Hcore_mat);
std::cout << "Here is the matrix Hcore:\n" << Hcore_mat << std::endl;
	h_inteval = 0;

//
// compute 2-e Coulomb integrals
//
	Eigen::MatrixXd Heffective_mat(nbasis * nbasis, nbasis * nbasis);
	Ref<TwoBodyInt> twoecoulomb_inteval = integral->electron_repulsion();
	get_two_electron_ints(twoecoulomb_inteval, Heffective_mat);
	twoecoulomb_inteval = 0;
	std::cout << "Here is the matrix Hcore:\n" << Heffective_mat << std::endl;

	integral = 0;

	std::size_t nocc = (mol->nuclear_charge()) / 2;
	const double nuc_repulsion = obs->molecule()->nuclear_repulsion_energy();

	Eigen::MatrixXd F(nbasis, nbasis);
	Eigen::MatrixXd X(nbasis, nbasis);

	//std::cout << Hcore_mat << std::endl;

	Eigen::MatrixXd C = HartreeFock(S_mat, Hcore_mat, Heffective_mat, nbasis,
			nocc, nuc_repulsion, F, X);

	TensorRank4 orb = IntegralTransformation(Heffective_mat, C, nbasis);

	TensorRank4 dij(nbasis, nbasis, nbasis, nbasis);
	TensorRank4 aij(nbasis, nbasis, nbasis, nbasis);

	//TensorRank4 Tij = MP2_second_quantization(orb, nocc, nbasis, F, X, dij,
		//aij);

	//MP2_linear_system_solver(orb, F, C, Tij, nocc, nbasis);

	//MP2_linear_system_solver_PNO(orb, F, C, Tij, nocc, nbasis, dij, aij);
	/*
	 MP2_linear_system_solver_truncated_PNO(orb, F, C, Tij, nocc, nbasis, dij,
	 aij);
	 */
	/*
	 CEPA_linear_solver(orb, F, C, Tij, nocc, nbasis);
	 */
	/*
	 LPNO_CEPA_linear_solver(orb, F, C, Tij, nocc, nbasis, dij);
	 */

	//CCSD_linear_solver(orb, F, C, Tij, nocc, nbasis);
	t_n = clock();
	std::cout << "time elapsed = " << double(t_n - t_0) / double(CLOCKS_PER_SEC)
			<< "\n";

	return 0;
}
