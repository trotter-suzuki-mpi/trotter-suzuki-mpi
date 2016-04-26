from __future__ import print_function, division
import numpy as np
import math

def vortex_position(grid, state, approx_cloud_radius=0.):
    """
    Get the position of a single vortex in the quantum state.
    
    Parameters
    ----------
    * `grid` : Lattice object
        Define the geometry of the simulation.
    * `state` : State object
        System's state.
    * `approx_cloud_radius` : float, optional
        Radius of the circle, centered at the Lattice's origin, where the vortex core
        is expected to be.
    
    Returns
    -------
    * `coords` numpy array
        Coordinates of the vortex core's position (coords[0]: x coordinate; coords[1]: y coordinate).
    
    Notes
    -----
    Only one vortex must be present in the state.
    
    Example
    -------
    
        >>> import trottersuzuki as ts  # import the module
        >>> import numpy as np
        >>> grid = ts.Lattice()  # Define the simulation's geometry
        >>> state = ts.GaussianState(grid, 1.)  # Create a state with gaussian wave function
        >>> def vortex_a(x, y):  # Define the vortex to be imprinted
        >>>     z = x + 1j*y
        >>>     angle = np.angle(z)
        >>>     return np.exp(1j * angle)
        >>> state.imprint(vortex)  # Imprint the vortex on the state
        >>> ts.vortex_position(grid, state)
        array([  8.88178420e-16,   8.88178420e-16])
    
    """
    if approx_cloud_radius == 0.:
        approx_cloud_radius = np.sqrt(2) * grid.length_x
    delta_y = grid.length_y / float(grid.dim_y)
    delta_x = grid.length_x / float(grid.dim_x)
    matrix = state.get_phase()
    # calculate norm gradient matrix
    norm_grad = np.zeros((grid.dim_x, grid.dim_y))
    for idy in range(1,grid.dim_y-1):
        for idx in range(1,grid.dim_x-1):
            if ((idx-grid.dim_x*0.5)**2 + (idy-grid.dim_y*0.5)**2) < approx_cloud_radius**2/delta_x**2:
                up = matrix[idy+1, idx]
                dw = matrix[idy-1, idx]
                rg = matrix[idy, idx+1]
                lf = matrix[idy, idx-1]

                if abs(up-dw) > np.pi:
                    up -= np.sign(up) * 2. * np.pi
                if abs(rg-lf) > np.pi:
                    rg -= np.sign(rg) * 2. * np.pi

                grad_x = (rg-lf)/(2.*delta_x)
                grad_y = (up-dw)/(2.*delta_y)
                norm_grad[idy, idx] = np.sqrt(grad_x**2 + grad_y**2)
    
    max_norm = np.nanmax(norm_grad)  # Find max value in norm_grad
    coord = np.transpose(np.where(norm_grad==max_norm))[0]
    
    # Check that the phase has a single discontinuity around the candidate vortex position
    def position(index, side):
        index = int(index)
        side = int(side)
        idx = int(math.fmod(index,4*side))  # position is periodic of period 4*radius
        quad = idx // side
        rest = int(math.fmod(idx,side))
        if quad == 0:
            x = - (side // 2 + 1)
            y = rest - side // 2
        if quad == 1:
            y = (side // 2 + 1)
            x = rest - side // 2
        if quad == 2:
            x = (side // 2 + 1)
            y = - (rest - side // 2)
        if quad == 3:
            y = - (side // 2 + 1)
            x = - (rest - side // 2)
        return np.array([y, x])
        
    side = 8 # must be an even number
    vortex = 0
    for i in range(0, side*4):  # Count the number of discontinuity in the phase pattern around the candidate vortex position
        pos1 = coord + position(i, side)
        pos2 = coord + position(i+1, side)
        if pos1[0] < 0 or pos1[0] >= grid.dim_x or pos1[1] < 0 or pos1[1] >= grid.dim_x:
            if pos2[0] < 0 or pos2[0] >= grid.dim_x or pos2[1] < 0 or pos2[1] >= grid.dim_x:
                break
        phase1 = matrix[pos1[0], pos1[1]]
        phase2 = matrix[pos2[0], pos2[1]]
        if np.abs(phase1 - phase2) >= np.pi * 1.7: 
            vortex += 1
    
    if vortex != 1:  # around the vortex there must be a single discontinuity in the phase
        return np.array([np.nan, np.nan])
    
    coord_x = []
    coord_y = []
    for idy in range(1,grid.dim_y-1):
        for idx in range(1,grid.dim_x-1):
            if norm_grad[idy, idx] >= max_norm*0.9:
                coord_x.append((idx + 0.5) * delta_x - 0.5 * grid.length_x)
                coord_y.append((idy + 0.5) * delta_y - 0.5 * grid.length_y)
                
    coords = np.zeros(2)
    for i in range(0, len(coord_x)):
        coords[1] += coord_y[i] / float(len(coord_x))
        coords[0] += coord_x[i] / float(len(coord_x))
        
    return coords
