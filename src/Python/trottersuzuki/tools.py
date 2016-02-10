import numpy as np

def vortex_position(grid, state, approx_cloud_radius=0.):
    """
    Get the position of a single vortex in the state.
    
    Parameters
    ----------
    * `grid` : Lattice object
        Define the geometry of the simulation.
    * 'state' : State object
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
        >>> from tools import vortex_position  # import function
        >>> grid = ts.Lattice()  # Define the simulation's geometry
        >>> state = ts.GaussianState(grid, 1.)  # Create a state with gaussian wave function
        >>> def vortex_a(x, y):  # Define the vortex to be imprinted
        >>>     z = x + 1j*y
        >>>     angle = np.angle(z)
        >>>     return np.exp(1j * angle)
        >>> state.imprint(vortex)  # Imprint the vortex on the state
        >>> vortex_position(grid, state)
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
    
    max_norm = np.nanmax(norm_grad)
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