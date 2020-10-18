import numpy as np

# Check https://www.desmos.com/calculator/gym5ie4g3w?lang=sv-SE
# for Flower inspiration

FLOWER_1 = {
    'description': "Pointy flower with 20 leaves",
    'func': eval("lambda theta: 3*np.sin(10*theta)"),
    'scale': 4.5,
    'lower_limit': 0,
    'upper_limit': 2 * np.pi
}

FLOWER_2 = {
    'description': "Flower with three leaf layers",
    'func': eval("lambda theta: 1.7 + 2.9*np.cos(-5.2*theta)**3"),
    'scale': 3.,
    'lower_limit': 0,
    'upper_limit': 12 * np.pi
}

FLOWER_3 = {
    'description': "Flower with round, overlapping leaves",
    'func': eval("lambda theta: np.sin(np.pi*theta)"),
    'scale': 10.,
    'lower_limit': 0,
    'upper_limit': 14 * np.pi
}

SPIRAL_1 = {
    'description': "Spiral with edgy, unstructured oscillations",
    'func': eval("lambda theta: theta + (0.4/10)*theta*np.sin(20*theta)"),
    'scale': 0.1,
    'lower_limit': 0,
    'upper_limit': 42 * np.pi
}

SPIRAL_2 = {
    'description': "Spiral with edgy, overlapping, unstructured oscillations",
    'func': eval("lambda theta: theta + (1.5/10)*theta*np.sin(20*theta)"),
    'scale': 0.1,
    'lower_limit': 0,
    'upper_limit': 38 * np.pi
}

HAMPA_1 = {
    'description': "Cannabis leaf",
    'func': eval("lambda theta: "
                 "(1 + (9/10)*np.cos(8*theta)) *"
                 "(1 + (1/10)*np.cos(24*theta)) *"
                 "(9/10 + (1/10)*np.cos(200*theta)) *"
                 "(1 + np.sin(theta))"),
    'scale': 3.,
    'lower_limit': 0,
    'upper_limit': 2 * np.pi
}

ERASE_1 = {
    'description': "Spiral that erases the current art",
    'func': eval("lambda theta: theta + np.cos(15)"),
    'scale': 0.15,
    'lower_limit': 0,
    'upper_limit': 29 * np.pi
}
