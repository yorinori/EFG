#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: leandro

GaussCellPoints


1)  nodes
:: number of nodes
:: coordinate of nodes



:: 

2) background cells
:: number of cells
:: coordinates of vertices 
:: connectivity of points 
:: 
  
  
3) Parametric space
:: vertices of parametric space
:: 

"""


def CellGaussPoints(xe_in, ND):
  """
  Parameters
  ----------
  xe_in : TYPE
    DESCRIPTION.
  ND : TYPE
    DESCRIPTION.

  Returns
  -------
  None.
  """
  import numpy as np
  # gpos    :: posições dos pontos de gauss dentro do espaço paramétrico em uma 
  #            e utilizando 4 pontos por dimensão
  # gweight :: respectivos pesos para os pontos de gauss
  #
  ngauss1D = 4
  gauss_position    = np.array([-.86113, -.33998,  .33998,  .86113 ])
  gauss_weight = np.array([ .34785,  .65214,  .65214,  .34785 ])

  # Definindo as coordenadas (xv) dos vértices do espaço paramétrico
  # (-1,-1), (1, -1), (1,1) , (-1, 1)
  xv = np.array([ (-1., -1.), (1., -1.), (1., 1.), (-1., 1.) ])
  
  # Trabalhando variáveis de entrada (específico para problemas 2D)
  
  xe, ye = np.hsplit( xe_in, ND )
  xe = xe.ravel()
  ye = ye.ravel()
  
  # Initializating Output variables
  xg       = np.zeros( ngauss1D*ngauss1D )
  yg       = np.zeros( ngauss1D*ngauss1D )
  weight   = np.zeros( ngauss1D*ngauss1D )
  jacobian = np.zeros( ngauss1D*ngauss1D )
  
  index = 0 
  for j in range(0, ngauss1D):  
    for i in range(0, ngauss1D): 
      # setting the coordinate of gauss points in the parametric space
      eta = gauss_position[j]
      ksi = gauss_position[i]
      
      N = np.zeros(4)      
      dNksi = np.zeros(4)   
      dNeta = np.zeros(4)
      
      # Loop over background points(vertices) in the parametric space
      for iv in range(0,4): 
        # vertices coordinates in the parametric space
        ksiJ = xv[iv][0]
        etaJ = xv[iv][1]
        
        # Função de forma linear da família serendipity e suas derivadas 
        N[iv]     = .25 * ( 1. + ksi*ksiJ ) * ( 1. + eta*etaJ )
        dNksi[iv] = .25 * ksiJ * ( 1. + eta*etaJ )
        dNeta[iv] = .25 * etaJ * ( 1. + ksi*ksiJ )
      
      # Cálculo do Jacobiano
      xksi = 0.
      yksi = 0.
      xeta = 0.
      yeta = 0.
      for iv in range(0,4):
        xksi = xksi + dNksi[iv] * xe[iv]
        yksi = yksi + dNksi[iv] * ye[iv]
        xeta = xeta + dNeta[iv] * xe[iv]
        yeta = yeta + dNeta[iv] * ye[iv]
        
      jacobian[index] = xksi*yeta - xeta*yksi
      
      #
      x_aux = 0.
      y_aux = 0.
      for iv in range(0,4):
        x_aux = x_aux + N[iv] * xe[iv]
        y_aux = y_aux + N[iv] * ye[iv]
      
      xg[index] = x_aux
      yg[index] = y_aux
      weight[index] = gauss_weight[i] * gauss_weight[j]
      
      index = index + 1
      
  return xg, yg, weight, jacobian



  
