import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Set, Tuple, Iterable
import networkx as nx
import ast
from scipy.ndimage import binary_erosion
from CGAL.CGAL_Kernel import Point_3, Vector_3, Point_2
from CGAL.CGAL_Point_set_3 import Point_set_3
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3, Polyhedron_3_Halfedge_handle, \
    Polyhedron_3_Vertex_handle, Polyhedron_3_Facet_handle, \
    Polyhedron_3_Halfedge_around_facet_circulator, Polyhedron_3_Edge_iterator, \
    Polyhedron_3_Halfedge_around_vertex_circulator
from CGAL.CGAL_Polygon_mesh_processing import Polylines, \
    remove_connected_components, keep_connected_components, area
import json
from scipy.ndimage.filters import median_filter
from tifffile import imsave, imread


Correspondence = Dict[str, Point_3]
ReverseCorrespondence = Dict[str, List[Point_3]]
Segmentation = Set[str]


def load_tif(filename: str) -> np.ndarray:
    image = imread(filename)
    image = np.moveaxis(image, 0, -1)

    if image.dtype == "uint16":
        alpha = 255 / 65535
        image = image.astype("float") * alpha
        image = np.round(image).astype("uint8")

    return image



## Balance the classes to the same size
def get_equal_grouping(merged_grouping):
    smallest_group = min([len(value) for value in  merged_grouping.groups.values()])
    ## Smallest group/ total number of x group
    fraction_of_spines = {key: smallest_group / len(value) for (key, value) in merged_grouping.groups.items()}
    equal_grouping = merged_grouping.get_balanced_subset(fraction_of_spines)
    return equal_grouping

