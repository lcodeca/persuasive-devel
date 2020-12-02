#!/usr/bin/env python3

""" Process the RLLIB metrics_XYZ.json """

import json
import os
import re

from copy import deepcopy

from numpyencoder import NumpyEncoder

class GenericGraphMaker():

    def __init__(self, input_dir, output_dir, filename, default):
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._dataset_fname = os.path.join(self._output_dir, filename)
        self._aggregated_dataset = deepcopy(default)

    @staticmethod
    def alphanumeric_sort(iterable):
        """
        Sorts the given iterable in the way that is expected.
        See: https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python
        """
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(iterable, key=alphanum_key)

    def generate(self):
        self._load_aggregate_data()
        self._process_and_save_new_metrics()
        self._generate_graphs()

    def _load_aggregate_data(self):
        if os.path.isfile(self._dataset_fname):
            with open(self._dataset_fname, 'r') as jsonfile:
                self._aggregated_dataset = json.load(jsonfile)
        # pprint(self._aggregated_dataset)

    def _aggregate_metrics(self, files):
        raise NotImplementedError()

    def _find_last_metric(self):
        raise NotImplementedError()

    def _process_and_save_new_metrics(self):
        # based on the aggregation, find the LAST_METRIC
        _last_metric_number = self._find_last_metric()
        # based on the input folder, load and process the NEW_METRICS
        files = self.alphanumeric_sort(os.listdir(self._input_dir))[_last_metric_number:]
        self._aggregate_metrics(files)
        # save the new aggregation to file
        with open(self._dataset_fname, 'w') as jsonfile:
            json.dump(self._aggregated_dataset, jsonfile, indent=2, cls=NumpyEncoder)

    def _generate_graphs(self):
        raise NotImplementedError()
