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
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression
from ipywidgets import widgets    
import csv
import random
import warnings
from histogram import histogram_intersection, histogram_intersection_pairwise, combined_kernel
warnings.filterwarnings(action='ignore', category=FutureWarning)
from spine_fitter import SpineGrouping   

Correspondence = Dict[str, Point_3]
ReverseCorrespondence = Dict[str, List[Point_3]]
Segmentation = Set[str]



## Balance the classes to the same size
def get_equal_grouping(merged_grouping):
    smallest_group = min([len(value) for value in  merged_grouping.groups.values()])
    ## Smallest group/ total number of x group
    fraction_of_spines = {key: smallest_group / len(value) for (key, value) in merged_grouping.groups.items()}
    equal_grouping = merged_grouping.get_balanced_subset(fraction_of_spines)
    return equal_grouping

def evaluate_classifier(label, metrics, ratios, merged_grouping, scores, kernel="rbf", verbose=True):
    metrics = metrics.get_spines_subset(merged_grouping.samples)
    
    output = []
    pred_groupings = []
    for train_ratio in ratios:
        # Select a balanced fraction of labeled spines for training; train_ratio controls its size.
        train_grouping = merged_grouping.get_balanced_subset(train_ratio)
        # Keep feature rows for those training spines, with names defining their row order.
        # Train metric and spine names
        train_metrics = metrics.get_spines_subset(train_grouping.samples)
        # Spine ID's
        train_names = train_metrics.ordered_spine_names
        
        # Hold out the other labeled spines so the model can be evaluated on unseen examples.
        # Test metrics and spine names
        test_grouping = merged_grouping.get_spines_subset(merged_grouping.samples.difference(train_grouping.samples))
        test_metrics = metrics.get_spines_subset(test_grouping.samples)
        # Spine ID's
        test_names = test_metrics.ordered_spine_names


        # String class label for each training spine Id
        train_target = [train_grouping.get_group(name) for name in train_names]
        # String class label for each testing spine Id
        # These held-out ground-truth labels are used to measure accuracy, not to fit the SVM.
        test_target = [test_grouping.get_group(name) for name in test_names]

        parameters = {'C': [1, 10]}

        train_data = train_metrics.as_array()
        test_data = test_metrics.as_array()

        svc = svm.SVC(kernel=kernel)
        clf = GridSearchCV(svc, parameters, cv=4)
        clf.fit(train_data, train_target)
         ## output score, determined by ground truth labels
        output.append(clf.score(test_data, test_target))

        # Run predict on every segmented spine
        pred_labels = clf.predict(metrics.as_array())
        # 3 empty sets, ensure every unique spine has a predicted group 
        pred_groups = {label: set() for label in ["Stubby", "Mushroom", "Thin"]}
        # Add spine id to its predicted group based on the predicted label
        for i, name in enumerate(metrics.ordered_spine_names):
            pred_groups[pred_labels[i]].add(name)
        
        pred_grouping = SpineGrouping(metrics.spine_names, pred_groups)
        ## Add this ratios prediction groupings
        pred_groupings.append(pred_grouping)
        
    reg = LinearRegression().fit(np.reshape(ratios, (-1, 1)), output)
    if verbose:
        plt.plot(ratios, output, label=label)
        print(f"{label} max accuracy: {max(output)}")

    # which grouping corresponds to the highest scoring training ratio
    best_grouping = pred_groupings[np.argmax(output)]
    # Scores at each training ratio
    scores[label] = (output)
    
    return best_grouping, output