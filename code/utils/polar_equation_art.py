import numpy as np

# Check https://www.desmos.com/calculator/gym5ie4g3w?lang=sv-SE
# for Flower inspiration

FLOWER_1 = {
    'func': eval("lambda theta: 3*np.sin(10*theta)"),
    'scale': 4.5,
    'lower_limit': 0,
    'upper_limit': 2 * np.pi
}

FLOWER_2 = {
    'func': eval("lambda theta: 1.7 + 2.9*np.cos(-5.2*theta)**3"),
    'scale': 3.,
    'lower_limit': 0,
    'upper_limit': 12 * np.pi
}

SPIRAL_1 = {
    'func': eval("lambda theta: 5 + np.sin(8*theta/5)"),
    'scale': 1.,
    'lower_limit': 0,
    'upper_limit': 10 * np.pi
}

HAMPA_1 = {
    'func': eval("lambda theta: "
                 "(1 + (9/10)*np.cos(8*theta)) *"
                 "(1 + (1/10)*np.cos(24*theta)) *"
                 "(9/10 + (1/10)*np.cos(200*theta)) *"
                 "(1 + np.sin(theta))"),
    'scale': 3.,
    'lower_limit': 0,
    'upper_limit': 2 * np.pi
}

ROSE = {
    'func': eval("lambda theta: np.sin(np.pi*theta)"),
    'scale': 10.,
    'lower_limit': 0,
    'upper_limit': 14 * np.pi
}

BUTTERFLY_1 = {
    'func': eval("lambda theta: 4*np.cos(10*np.cos(theta))"),
    'scale': 1.,
    'lower_limit': 0,
    'upper_limit': 2 * np.pi
}