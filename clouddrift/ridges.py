import numpy as np

def ridgemap(
        grid_shape: int | tuple, 
        *ridge_data: np.ndarray,
        collapse: bool = False
) -> list:
    """
    Maps ridge quantities back onto the time series or grid.
    
    This function takes ridge quantities and their corresponding indices
    and maps them onto a grid of specified dimensions, used to
    reconstruct time series data from ridge-extracted information.
    
    Parameters
    ----------
    grid_shape : int or tuple
        Either a single integer representing the length of the time series (M),
        or a tuple/list (M, N) representing 2D grid dimensions.
    *ridge_data : np.ndarray
        A variable number of numpy arrays containing ridge quantities and indices.
        Format: X1R, X2R, ..., XPR, IR [, KR]
        - If grid_shape is a tuple: The last two arrays must be row indices (IR) and column indices (KR)
        - If grid_shape is a single value: The last array must be row indices (IR)
        - All arrays preceding indices are considered ridge quantities to be mapped
        
        If IR and XR contain multiple ridges separated by NaNs (as output by
        ridge detection algorithms), the result will have each ridge in a separate column.
    collapse : bool, optional
        If True, combines values from all ridges into a single value per grid point
        by summing the first quantity and averaging the rest. Default is False.
    
    Returns
    -------
    list
        A list of NumPy arrays containing the mapped ridge quantities and multiplicity.
        - Each mapped quantity is an array of shape (M, N) where N=1 if not specified
        - The last element is the multiplicity (count of finite values) for each row
        - If collapse=True, only one ridge quantity array is returned (plus multiplicity)
    
    Notes
    -----
    - Indices in IR/KR are expected to be 1-indexed (converted to 0-indexed internally)
    - Values not specified by the indices are filled with NaNs
    - The ridge multiplicity is the number of ridges present at each time/location
    
    Examples
    --------
    # Map a single ridge quantity
    x = ridgemap(M, xr, ir)
    
    # Map multiple ridge quantities
    [x, f] = ridgemap(M, xr, fr, ir)
    
    # Collapse multiple ridges into a single time series
    [x, f] = ridgemap(M, xr, fr, ir, collapse=True)
    
    # Get ridge multiplicity along with mapped quantities
    [x, f, mult] = ridgemap(M, xr, fr, ir)
    """
    # Determine grid dimensions
    if isinstance(grid_shape, (list, tuple)):
        num_rows, num_cols = grid_shape
        has_col_dimension = True
    else:
        num_rows, num_cols = grid_shape, 1
        has_col_dimension = False

    # Extract indices and ridge quantities from arguments
    if has_col_dimension and len(ridge_data) >= 2:
        col_indices = ridge_data[-1]
        row_indices = ridge_data[-2]
        ridge_quantities = ridge_data[:-2]
    else:
        row_indices = ridge_data[-1]
        col_indices = None
        ridge_quantities = ridge_data[:-1]

    # Handle empty or None indices
    if row_indices is None or len(np.atleast_1d(row_indices)) == 0:
        mapped_quantities = [np.full((num_rows, 1), np.nan) for _ in ridge_quantities]
        multiplicity = np.zeros((num_rows, 1))
    else:
        row_indices = np.atleast_2d(row_indices).astype(float)
        mapped_quantities = []
        
        for quantity in ridge_quantities:
            quantity = np.atleast_2d(quantity).astype(float)
            mapped_quantity = np.full((num_rows, 1), np.nan)
            
            # Advanced indexing to place values at their corresponding grid positions
            valid_mask = np.isfinite(row_indices)
            mapped_rows = (row_indices[valid_mask] - 1).astype(int)  # Convert to 0-indexed
            mapped_quantity[mapped_rows, 0] = quantity[valid_mask]
            
            mapped_quantities.append(mapped_quantity)
            
        # Count valid entries per row
        multiplicity = np.sum(np.isfinite(mapped_quantities[0]), axis=1, keepdims=True)

    # Optionally combine all ridge quantities
    if collapse and mapped_quantities:
        combined = mapped_quantities[0].sum(axis=1, keepdims=True)
        for quantity in mapped_quantities[1:]:
            combined += quantity.mean(axis=1, keepdims=True)
        mapped_quantities = [combined]

    # Add multiplicity as the last element
    mapped_quantities.append(multiplicity)

    return mapped_quantities