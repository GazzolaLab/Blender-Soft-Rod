import numpy as np


def _validate_position(position: NDArray) -> None:
    """
    Checks if inputted position values are valid

    Paramters
    ---------
    position: NDArray
        Position input (endpoint or centerpoint depending on Object type)

    Raises
    ------
    ValueError
        If the position is the wrong shape or contains NaN values
    """

    if position.shape[0] != 3:
        raise ValueError("The shape of the position is incorrect.")
    if np.isnan(position).any():
        raise ValueError("The position contains NaN values.")


def _validate_radius(radius: float) -> None:
    """
    Checks if inputted radius value is valid

    Parameters:
    -----------
    radius: Float
        Radius input

    Raises
    ------
    ValueError
        If the radius is not positive, or contains NaN values
    """

    if not isinstance(radius, Number) or radius <= 0:
        raise ValueError("The radius must be a positive float.")
    if np.isnan(radius):
        raise ValueError("The radius contains NaN values.")


def _validate_radii(radii: NDArray) -> None:
    """
    Checks if inputted radius value is valid

    Parameters:
    -----------
    radii: NDArray
        Radius input

    Raises
    ------
    ValueError
        If the radius is not positive, or contains NaN values
    """

    if (radii <= 0).any():
        raise ValueError("The radius must be a positive float.")
    if np.isnan(radius).any():
        raise ValueError("The radius contains NaN values.")
