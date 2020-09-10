# -*- coding: utf-8 -*-
"""
Spyder Editor


ref de número zero = 1.73864796e-18 ? 
"""

def quartic_spline(x,xi,ds):
  """
  Parameters
  ----------
  x : float
    point of interest (often a gauss point).
  xi : float
    coordinate of node (1D).
  ds : float
    lenght of support domain 1D.

  Returns
  -------
  w    : float
       weight function 1D
  dwdx : float
       first derivative of weight function 1D
  """
  import numpy as np
     
  r = np.abs(x - xi)/ds
  if( r <= 1. ):
    w = 1. - 6.*( r )**2. + 8.*( r )**3. - 3.*( r )**4.   
    if( x >= xi ):
      dwdx = ( -12.*r + 24.*( r )**2. -12.*( r )**3. ) * (1./ds) 
    if( xi > x):
      dwdx = ( -12.*r + 24.*( r )**2. -12.*( r )**3. ) * (-1./ds) 
    return w, dwdx
  else: # r > 1.
    w = 0.
    dwdx = 0.
    return w, dwdx

def quartic_spline2D(x, y, xi, yi, dsx, dsy):
    wx, dwdx = quartic_spline(x, xi, dsx)
    wy, dwdy = quartic_spline(y, yi, dsy)
    
    wi = wx * wy
    dwidx = dwdx * wy
    dwidy = wx * dwdy
    return wi, dwidx, dwidy
#
#
#
def basis( x, y ):
  """
  Parameters
  ----------
  x : float
    coordinate x.
  y : float
    coordinate y.
  Returns
  -------
  p    : np.array
       column vector of basis monomials applied to (x, y)
  dpdx : np.array
       first partial derivative in x of basis monomials applied to (x, y)
  dpdy : np.array
       first partial derivative in y of basis monomials applied to (x, y)
  """
  import numpy as np
  
  p    = np.array([ [ 1.],
                    [ x ],
                    [ y ]  ])

  dpdx = np.array([ [ 0.],
                    [ 1.],
                    [ 0.]  ])
  
  dpdy = np.array([ [ 0.],
                    [ 0.],
                    [ 1.]  ])
    
  return p, dpdx, dpdy
#
#  
#
def MLS_ShapeFunc(poi, xi_in, n, dsx, dsy):
  import numpy as np
  from numpy.linalg import inv
  m = 3 #Dimension of basis functions
  
  B = np.array( [] ).reshape(m,0) #Dimensão deve ser conhecida, 3 é a dimensão da base
  dBdx = np.array( [] ).reshape(m,0) ;  dBdy = np.array( [] ).reshape(m,0)
  A = np.zeros( (m,m) ) ; dAdx = np.zeros( (m,m) ); dAdy = np.zeros( (m,m) )
  for i in range(n): 
    xi = xi_in[i]    
    wi, dwidx, dwidy = quartic_spline2D(poi[0], poi[1], xi[0], xi[1], dsx[i], dsy[i])

    # p(xi)
    p_xi = np.array( [ [  1.   ],
                       [ xi[0] ],
                       [ xi[1] ]  ] )

    ppT = np.matmul(p_xi, p_xi.T)     #ppT = p_xi * p_xi.T
    A = A + wi * ppT
    dAdx = dAdx + dwidx * ppT
    dAdy = dAdy + dwidy * ppT

    B = np.hstack( [B, wi * p_xi] )
    dBdx = np.hstack( [dBdx, dwidx * p_xi] )
    dBdy = np.hstack( [dBdy, dwidy * p_xi] )

  # p(x)
  """p_x = np.array([ [  1.   ],
                   [ point[0] ],
                   [ point[1] ]  ])
  """
  #p(x), dpdx(x), dpdy(x)
  p_x, dpdx, dpdy = basis( poi[0], poi[1] )
  #
  phi = np.matmul( p_x.T, np.matmul( inv(A), B ) )  # p_x.T * inv(A) * B
  #
  # Cálculo de gamma
  #
  gamma = np.matmul( inv(A), p_x ) 
  #
  # Cálculo da derivada de gamma em relação a x 
  #
  aux_x = dpdx - np.matmul( dAdx, gamma )
  dgdx = np.matmul ( inv(A), aux_x  ) 
  #
  # Cálculo da derivada de gamma em relação a y
  #
  aux_y = dpdy - np.matmul( dAdy, gamma )
  dgdy = np.matmul ( inv(A), aux_y )
  #
  # Cálculo de dphi/dx e dphi/dy
  #
  dphidx = np.matmul(dgdx.T, B) + np.matmul(gamma.T, dBdx)

  dphidy = np.matmul(dgdy.T, B) + np.matmul(gamma.T, dBdy)
  #
  # Transform outputs in 1D ndarray
  #
  phi = phi.ravel()
  dphidx = dphidx.ravel()
  dphidy = dphidy.ravel()
  #
  return phi, dphidx, dphidy
