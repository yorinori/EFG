#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module input for MFree_EFG

"""
import numpy as np
"""
Input Parameters

"""
L = 48.0           # The Length of the beam
H = 12.0           # The Height of the beam
E = 0.300E+08      # Young's modulus
poisson = 0.30000  # Poisson's ratio
P = 1000.00000     # Loading (integration of the distributed traction)

nx = 24                  # number of divisions in x direction
ny = 6                   # number of divisions in y direction
numNode = (nx+1)*(ny+1)  # (175) Total number of field nodes

nx_BKC = 10       # number of divisions in x direction for background cells
ny_BKC = 4        # number of divisions in y direction for background cells

numBKV = (nx_BKC+1) * (ny_BKC+1)  # (55) number of background vertices
numBKC = nx_BKC * ny_BKC          # (40) number of background cells

ngauss1D = 4                    # number of gauss point in 1D
ngauss2D = ngauss1D * ngauss1D  # total number of gauss points in 2D

alfs = 3.0                      # influence domain parameter
ND = 2                          # Number of dimensions
"""

Calculate nodes positions for the beam
---------------------------------------
The field nodes positions are regularly distributed.

x_node :: nd.array(:,2) with x coordinates in the first column and
         y coordinates in the second column
  x_node[:][0] :: nd.array with x coordinates of field nodes
  x_node[:][1] :: nd.array with y coodinates of field nodes

"""
x_aux = np.linspace(0., L, nx+1)
y_aux = np.linspace(H/2., -H/2., ny+1)

x_aux, y_aux = np.meshgrid(x_aux, y_aux)

x_aux = np.concatenate(np.hsplit(x_aux, nx+1))
y_aux = np.concatenate(np.hsplit(y_aux, nx+1))  # it's nx. It's right

x_node = np.hstack((x_aux, y_aux))

"""
# Former approach

index = 0
x_node = np.zeros(x_aux.size)
y_node = np.zeros(y_aux.size)
for i in range(0,25):
  for j in range(0,7):
    x_node[index] = x_aux[j][i]
    y_node[index] = y_aux[j][i]
    index = index + 1
"""


"""

Background Cells
------------------
The background cells have rectangular form.

x_BKC :: nd.array(:,2) with x coordinates in the first column and y coordinates
         in the second column
  x_BKC[:][0] :: nd.array with x coordinates of the background cells points
  x_BKC[:][1] :: nd.array with y coordinates of the background cells points

V4Cell :: Vertices for Background Cells (Connectivity)
          The vertices are displayed in counterclockwise beginning with the
          left-upper vertice
"""
x_aux = np.linspace(0., L, nx_BKC+1)
y_aux = np.linspace(H/2., -H/2., ny_BKC+1)

x_aux, y_aux = np.meshgrid(x_aux, y_aux)

x_aux = np.concatenate(np.hsplit(x_aux, nx_BKC+1))
y_aux = np.concatenate(np.hsplit(y_aux, nx_BKC+1))  # it's nx. It's right

x_BKC = np.hstack((x_aux, y_aux))

"""
#Former approach

index = 0
x_BC = np.zeros(x_aux.size)
y_BC = np.zeros(y_aux.size)
for i in range(0,nx_BC+1):
  for j in range(0,ny_BC+1):
    print( 'i=', i, 'j=', j, '  coord =(',x_aux[j][i],' , ', y_aux[j][i], ')   index=',index )
    x_BC[index] = x_aux[j][i]
    y_BC[index] = y_aux[j][i]
    index = index + 1
"""
# Setting vertices for each background cell
V4Cell = []
for i in range(nx_BKC):
    for j in range(ny_BKC):
        #  print('i=', i, 'j=', j)  # , 'celula=',index, ' nodes=',
        #          i*ndivyq+j , i*ndivyq+j+1, (i+1)*ndivyq+j+1, (i+1)*ndivyq+j)
        V4Cell.append([i*(ny_BKC+1)+j, i*(ny_BKC+1)+j+1,
                       (i+1)*(ny_BKC+1)+j+1, (i+1)*(ny_BKC+1)+j])


"""
Boundary Conditions
--------------------

"""

# Natural BC distributed

NBC_distributed = [[51, 50], [52, 51], [53, 52], [54, 53]]

# Essential BC

NodeEBC = [0, 1, 2, 3, 4, 5, 6]

EBC = [[True, -0.00000E-25, True, -0.60000E-04],
       [True, -0.70988E-05, True, -0.26667E-04],
       [True, -0.56790E-05, True, -0.66667E-05],
       [True,  0.00000E-25, True,  0.00000E-25],
       [True,  0.56790E-05, True, -0.66667E-05],
       [True,  0.70988E-05, True, -0.26667E-04],
       [True,  0.00000E-25, True, -0.60000E-04]]

pAlf = 100000000.000000

# Natural BC concentrated

NodeNBC_concentrated = [168, 169, 170, 171, 172, 173, 174]

NBC_concentrated = [[True, 0.0, True, 0.0],
                    [True, 0.0, True, 0.0],
                    [True, 0.0, True, 0.0],
                    [True, 0.0, True, 0.0],
                    [True, 0.0, True, 0.0],
                    [True, 0.0, True, 0.0],
                    [True, 0.0, True, 0.0]]
