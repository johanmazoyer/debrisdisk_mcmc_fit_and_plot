def au_to_mas(d_in_au, distance_star_in_pc):
    """convert a distance from au to milliarcseconds

    Args:
        d_in_au: distance in au
        distance_star_in_pc: distance to the star in pc

    Returns:
        distance in milliarcseconds
    """

    return d_in_au / distance_star_in_pc * 1000.


def mas_to_au(d_in_mas, distance_star_in_pc):
    """convert a distance from milliarcseconds to au

    Args:
        d_in_mas: distance in milliarcseconds
        distance_star_in_pc: distance to the star in pc

    Returns:
        distance in au
    """

    return d_in_mas * distance_star_in_pc / 1000.


def mas_to_pix(d_in_mas, pixscale):
    """convert a distance from milliarcseconds to pix

    Args:
        d_in_mas: distance in milliarcseconds
        pixscale: pixel scale in arcsecond per pixel

    Returns:
        distance in pixels
    """

    return d_in_mas * pixscale * 1000.


def pix_to_mas(d_in_pix, pixscale):
    """convert a distance from pix to milliarcseconds

    Args:
        d_in_pix: distance in pixels
        pixscale: pixel scale in arcsecond per pixel

    Returns:
        distance in milliarcseconds
    """

    return d_in_pix / pixscale


def pix_to_au(d_in_pix, pixscale, distance_star_in_pc):
    """convert a distance from pixels to au

    Args:
        d_in_pix: distance in pixels
        pixscale: pixel scale in arcsecond per pixel
        distance_star_in_pc: distance to the star in pc

    Returns:
        distance in au
    """

    return mas_to_au(pix_to_mas(d_in_pix, pixscale), distance_star_in_pc)


def au_to_pix(d_in_au, pixscale, distance_star_in_pc):
    """convert a distance in au to pixels

    Args:
        d_in_au: distance in au
        pixscale: pixel scale in arcsecond per pixel
        distance_star_in_pc: distance to the star in pc

    Returns:
        distance in pixels
    """

    return mas_to_pix(au_to_mas(d_in_au, distance_star_in_pc), pixscale)
