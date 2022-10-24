"""
Unit testing for WFPT likelihood function.

This code compares WFPT likelihood function with
old implementation of WFPT from (https://github.com/hddm-devs/hddm)
"""

import math
import os
import sys
import unittest

import hddm_wfpt
import numpy as np
import ssms.basic_simulators.simulator as simulator
from numpy.random import rand

sys.path.insert(0, os.path.abspath("src"))
from src.hssm.wfpt import decision_func, log_pdf_sv


class TestWfpt(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestWfpt, self).__init__(*args, **kwargs)
        self.n_samples = 500
        self.sim_out = simulator(
            theta=[0.0, 1.5, 0.5, 1.0], model="ddm", n_samples=self.n_samples
        )
        self.data_tmp = self.sim_out["rts"] * self.sim_out["choices"]
        self.err = 1e-7

    def test_kterm(self):
        """
        This function defines a range of kterms and tests results to
         makes sure they are not equal to infinity or unknown values
        """
        for k_term in range(1, 11):
            v = (rand() - 0.5) * 1.5
            sv = 0
            a = 1.5 + rand()
            z = 0.5 * rand()
            t = rand() * 0.5
            err = self.err
            logp = log_pdf_sv(
                self.data_tmp.flatten(), v, sv, a, z, t, err, k_terms=k_term
            )
            logp = logp.eval()
            np.testing.assert_equal(math.isfinite(logp), True)
            np.testing.assert_equal(math.isnan(logp), False)

    def test_decision(self):
        """
        This function tests output of decision function
        """
        decision = decision_func()
        lambda_rt = decision(self.data_tmp.flatten(), self.err)
        np.testing.assert_equal(all(v == False for v in lambda_rt.eval()), True)
        np.testing.assert_equal(len(self.data_tmp.flatten()), len(lambda_rt.eval()))

    def test_logp(self):
        """
        This function compares new and old implementation of logp calculation
        """
        for i in range(20):
            v = (rand() - 0.5) * 1.5
            sv = 0
            a = 1.5 + rand()
            z = 0.5 * rand()
            t = rand() * 0.5
            err = 1e-7
            aesara_log = log_pdf_sv(self.data_tmp.flatten(), v, sv, a, z, t, err=err)
            cython_log = hddm_wfpt.wfpt.pdf_array(
                self.data_tmp.flatten(), v, sv, a, z, 0, t, 0, err, 1
            ).sum()
            np.testing.assert_array_almost_equal(aesara_log.eval(), cython_log, 9)
