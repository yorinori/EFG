#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from numpy.linalg import inv
from gausscell import CellGaussPoints
from MLS_functions import MLS_ShapeFunc

# Input data (from input module)
from input import x_BKC, V4Cell, numNode, numBKC
from input import x_node
from input import L, H, nx, ny, ND
from input import alfs
from input import ngauss1D, ngauss2D
from input import E, poisson
from input import NBC_distributed, P
from input import NodeEBC, EBC, pAlf
from input import NodeNBC_concentrated, NBC_concentrated


def SupportDomain(x_poi, y_poi, dsx, dsy, x, y):
    """
    Parameters
    ----------
    x_poi : numpy.float64
      x coordinate of the point of interest (very often a Gauss point)
    y_poi : numpy.float64
      y coordinate of the point of interest (very often a Gauss Point)
    dsx : numpy.ndarray
      Support domain size in x direction for each node
    dsy : numpy.ndarray
      Support domain size in y direction for each node
    x   : numpy.array
      x coordinates of nodes
    y   : numpy.array
      y coordinates of nodes

    Returns
    -------
    mask    : numpy.ndarray
      A boolean mask with for nodes within the support domain.
    maskpos : numpy.ndarray
      A mask with the positions of nodes within the support domain.
    n       : int
      number of nodes within Support Domain
    """
    ref = 1.73864796e-18
    length = len(dsx)

    dx = dsx - np.abs(x_poi * np.ones(length) - x)
    dy = dsy - np.abs(y_poi * np.ones(length) - y)

    mask = (dx >= ref) & (dy >= ref)
    maskpos = np.where((dx >= ref) & (dy >= ref))[0]
    n = int(np.sum(mask))

    return mask, maskpos, n


#
xi_node, yi_node = np.hsplit(x_node, ND)
xi_node = xi_node.ravel()
yi_node = yi_node.ravel()

# Compute material matrix D for plane STRESS [FOR SLOPES USE PLANE STRAIN ]
# Plane stress state
D = np.array([[1.,      poisson,              0.],
              [poisson,      1.,              0.],
              [0.,           0., (1.-poisson)/2.]])

D = (E/(1. - poisson*poisson)) * D

"""
# Plane Strain state
Daux1 = ( E * (1. - poisson) )/( ( 1.+poisson )*( 1.-2.*poisson ) )
Daux2 = poisson/( 1.-poisson )
Daux3 = ( 1.-2.*poisson)/( 2.*( 1.-poisson ))

D = np.array([ [   1. , Daux2,   0. ],
               [ Daux2,   1. ,   0. ],
               [   0. ,   0. , Daux3] ])

D = Daux1 * D
"""

# Determine sizes of influence domains (Uniform nodal spacing)
xspace = L/nx
yspace = H/ny

dsx = np.ones(numNode) * alfs * xspace
dsy = np.ones(numNode) * alfs * yspace

# Global Stiff matrix and Force vector
Kglobal = np.zeros((2*numNode, 2*numNode))
force = np.zeros(2*numNode)

# --- Big Loop for background cells ---/
for ibc in range(numBKC):
    print("Background cell number: ", ibc)

    # Coordinates of background points in real space
    xe = x_BKC[V4Cell[ibc]]

    # --- Set Gauss points for this cell ---/
    x_gauss, y_gauss, weight, jac = CellGaussPoints(xe, ND)

    # --- Loop over Gauss points to assemble discrete equations ---/
    for igp in range(0, ngauss2D):

        # --- Determine the support domain of Gauss point ---/
        mask, maskpos, n = SupportDomain(
                        x_gauss[igp], y_gauss[igp], dsx, dsy, xi_node, yi_node)

        # --- Construct MLS shape functions for a Gauss point ---/

        poi = x_gauss[igp], y_gauss[igp]

        phi, dphix, dphiy = MLS_ShapeFunc(
                                    poi, x_node[mask], n, dsx[mask], dsy[mask])

        # --- Compute the Stiffness matrix for a Gauss point
        for i in range(n):
            for j in range(n):

                B_i = np.array([[dphix[i],       0.],
                                [0.,       dphiy[i]],
                                [dphiy[i], dphix[i]]])

                B_j = np.array([[dphix[j],       0.],
                                [0.,       dphiy[j]],
                                [dphiy[j], dphix[j]]])

                Kij = weight[igp] * jac[igp] * np.matmul(
                                                      B_i.T, np.matmul(D, B_j))

                I = maskpos[i]
                J = maskpos[j]

                Kglobal[2*I: ((2*I)+1)+1, 2*J: ((2*J)+1)+1] += Kij

        #
    # End of loop for gauss points
# End of loop for background cells

# --- Implement Distributed Natural BC

# Depois colocar isso em uma função
gauss_position = np.array([-.86113, -.33998, .33998, .86113])
gauss_weight = np.array([.34785, .65214, .65214, .34785])

for iNBC in range(len(NBC_distributed)):
    X1 = x_BKC[NBC_distributed[iNBC][0]]
    X2 = x_BKC[NBC_distributed[iNBC][1]]

    ax = 0.5 * (X1[0] - X2[0])
    ay = 0.5 * (X1[1] - X2[1])
    bx = 0.5 * (X1[0] + X2[0])
    by = 0.5 * (X1[1] + X2[1])

    for igp in range(ngauss1D):
        x_gpos = ax * gauss_position[igp] + bx
        y_gpos = ay * gauss_position[igp] + by
        jac = 0.5 * np.sqrt((X1[0] - X2[0])**2. + (X1[1] - X2[1])**2.)
        Imo = (1./12.) * H**3.
        Ty = (-P/(2.*Imo)) * (((H**2.)/4.) - (y_gpos**2.))

        # Support Domain
        mask, maskpos, n = SupportDomain(
                                    x_gpos, y_gpos, dsx, dsy, xi_node, yi_node)

        # Calculate MLS Shape functions
        poi = x_gpos, y_gpos  # Point of interest
        phi, dphix, dphiy = MLS_ShapeFunc(
                                    poi, x_node[mask], n, dsx[mask], dsy[mask])

        for i in range(n):
            I = maskpos[i]
            force[2*I+1] += gauss_weight[igp] * jac * phi[i] * Ty

# --- Essential Boundary Conditions

Kmax = Kglobal.diagonal().max()

for iEBC in range(len(NodeEBC)):
    x_poi = xi_node[NodeEBC[iEBC]]
    y_poi = yi_node[NodeEBC[iEBC]]

    mask, maskpos, n = SupportDomain(x_poi, y_poi, dsx, dsy, xi_node, yi_node)

    poi = x_poi, y_poi  # Point of interest
    phi, dphix, dphiy = MLS_ShapeFunc(
                                    poi, x_node[mask], n, dsx[mask], dsy[mask])

    for i in range(n):
        for j in range(n):
            I = maskpos[i]
            J = maskpos[j]
            if EBC[iEBC][0]:
                Kglobal[2*I, 2*J] += -pAlf * Kmax * phi[i] * phi[j]
            if EBC[iEBC][2]:
                Kglobal[2*I+1, 2*J+1] += -pAlf * Kmax * phi[i] * phi[j]

        I = maskpos[i]
        if EBC[iEBC][0]:
            uu = EBC[iEBC][1]
            force[2*I] += -pAlf * uu * Kmax * phi[i]
        if EBC[iEBC][2]:
            uu = EBC[iEBC][3]
            force[2*I+1] += -pAlf * uu * Kmax * phi[i]

# --- Natural BC concentrated
for iNBC in range(len(NodeNBC_concentrated)):
    x_poi = xi_node[NodeNBC_concentrated[iNBC]]
    y_poi = yi_node[NodeNBC_concentrated[iNBC]]

    mask, maskpos, n = SupportDomain(x_poi, y_poi, dsx, dsy, xi_node, yi_node)

    poi = x_poi, y_poi  # Point of interest
    phi, dphix, dphiy = MLS_ShapeFunc(
                                    poi, x_node[mask], n, dsx[mask], dsy[mask])

    for i in range(n):
        I = maskpos[i]
        if NBC_concentrated[iNBC][0]:
            uu = NBC_concentrated[iNBC][1]
            force[2*I] += phi[i] * uu
        if NBC_concentrated[iNBC][2]:
            uu = NBC_concentrated[iNBC][3]
            force[2*I+1] += phi[i] * uu

# --- Solve system

print("Resolvendo o sistema")
U = np.matmul(inv(Kglobal), force)

# --- Analytical solution

Ua_x = np.zeros(numNode)
Ua_y = np.zeros(numNode)
for i in range(numNode):
    x = xi_node[i]
    y = yi_node[i]
    aux1 = (P * y)/(6. * E * Imo)
    aux2 = (6. * L - 3. * x) * x
    aux3 = (2. + poisson) * ((y**2.) - ((H**2.)/(4.)))

    Ua_x[i] = aux1 * (aux2 + aux3)

    aux1 = -P/(6. * E * Imo)
    aux2 = (3. * poisson * y**2.) * (L - x)
    aux3 = (4. + 5. * poisson) * ((H**2. * x)/4.)
    aux4 = (3. * L - x) * x**2.

    Ua_y[i] = aux1 * (aux2 + aux3 + aux4)

# --- Get displacement

disp = np.zeros(2*numNode)
for iNode in range(numNode):
    x_poi = xi_node[iNode]
    y_poi = yi_node[iNode]

    mask, maskpos, n = SupportDomain(x_poi, y_poi, dsx, dsy, xi_node, yi_node)

    poi = x_poi, y_poi  # Point of interest
    phi, dphix, dphiy = MLS_ShapeFunc(
                                    poi, x_node[mask], n, dsx[mask], dsy[mask])

    for i in range(n):
        I = maskpos[i]
        disp[2*iNode] += phi[i] * U[2*I]
        disp[2*iNode+1] += phi[i] * U[2*I+1]

Ux = np.zeros(numNode)
Uy = np.zeros(numNode)
for iNode in range(numNode):
    print('{:5d},   {:+0.5e},   {:+0.5e}   {:+0.5e},   {:+0.5e} {:+0.5e},   {:+0.5e}'.format(
              iNode, Ua_x[iNode], U[2*iNode], disp[2*iNode], Ua_y[iNode], U[2*iNode+1],disp[2*iNode+1]))

    Ux[iNode] = xi_node[iNode] + disp[2*iNode]
    Uy[iNode] = yi_node[iNode] + disp[2*iNode+1]

# --- Get Stress

Dinv = inv(D)
enorm = 0.
# --- Compute energy error

# Big loop for background cells
for ibc in range(numBKC):

    # Coordinates of background points in real space
    xe = x_BKC[V4Cell[ibc]]

    # --- Set Gauss points for this cell ---/
    x_gauss, y_gauss, weight, jac = CellGaussPoints(xe, ND)

    # --- Loop over Gauss points to assemble discrete equations ---/
    for igp in range(0, ngauss2D):
        Stress = np.zeros(3)
        Stress_exact = np.zeros(3)
        # --- Determine the support domain of Gauss point ---/
        mask, maskpos, n = SupportDomain(
                        x_gauss[igp], y_gauss[igp], dsx, dsy, xi_node, yi_node)

        # --- Construct MLS shape functions for a Gauss point ---/

        poi = x_gauss[igp], y_gauss[igp]

        phi, dphix, dphiy = MLS_ShapeFunc(
                                    poi, x_node[mask], n, dsx[mask], dsy[mask])

        # --- Compute the Stiffness matrix for a Gauss point
        Bmat = np.array([]).reshape(3, 0)
        uu = np.array([]).reshape(0)
        for i in range(n):

            B_i = np.array([[dphix[i],       0.],
                            [0.,       dphiy[i]],
                            [dphiy[i], dphix[i]]])

            Bmat = np.hstack([Bmat, B_i])
            I = maskpos[i]
            uu = np.hstack([uu, disp[2*I: (2*I+1)+1]])

        uu = uu.reshape(2*n, 1)
        Stress = np.matmul(D, np.matmul(Bmat, uu))

        # --- Exact stress for beam problem
        Stress_exact[0] = (1./Imo) * P * (L - poi[0]) * poi[1]
        Stress_exact[1] = 0.
        Stress_exact[2] = -0.5 * (P/Imo) * (0.25 * (H**2.) - poi[1]**2.)

        err = Stress.ravel() - Stress_exact

        der = np.zeros(3)
        for jer in range(3):
            for ker in range(3):
                der[jer] += Dinv[jer, ker] * err[ker]

        err2 = 0.
        for mer in range(3):
            err2 += weight[igp] * jac[igp] * (0.5 * der[mer] * err[mer])
        enorm += err2
    # End of Loop for gauss points
# End of Big Loop for background cells
enorm = np.sqrt(enorm)
