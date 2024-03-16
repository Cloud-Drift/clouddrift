import unittest

import numpy as np

from clouddrift.transfer import (
    transfer_function,
)

if __name__ == "__main__":
    unittest.main()

class transfer_test_gradient(unittest.TestCase):
    delta = 10**np.arange(-1,0.05,3)
    bld = 10**np.arange(np.log10(15.15),5,0.05)
    [delta_grid,bld_grid] = np.meshgrid(delta,bld)

    def test_gradient(self):
        # Test the gradient of the transfer function
        omega = np.array([1e-4])
        z = 15
        cor_freq = 1e-4
        mu = 0
        delta_delta = 1e-6
        delta_bld = 1e-6
        # initialize the transfer function
        transfer_function_0 = np.zeros((len(self.delta), len(self.bld)), dtype=complex)
        transfer_function_1 = np.zeros((len(self.delta), len(self.bld)), dtype=complex)
        transfer_function_2 = np.zeros((len(self.delta), len(self.bld)), dtype=complex)
        transfer_function_3 = np.zeros((len(self.delta), len(self.bld)), dtype=complex)
        transfer_function_4 = np.zeros((len(self.delta), len(self.bld)), dtype=complex)

        for i in range(len(self.delta)):
            for j in range(len(self.bld)):
                transfer_function_0[i,j] = transfer_function(
                    omega=omega,
                    z=z,
                    cor_freq=cor_freq,
                    delta=self.delta[i],
                    mu=mu,
                    bld=self.bld[j],
                )
                transfer_function_1[i,j] = transfer_function(
                    omega=omega,
                    z=z,
                    cor_freq=cor_freq,
                    delta=self.delta[i]+delta_delta/2,
                    mu=mu,
                    bld=self.bld[j],
                )
                transfer_function_2[i,j] = transfer_function(
                    omega=omega,
                    z=z,
                    cor_freq=cor_freq,
                    delta=self.delta[i]-delta_delta/2,
                    mu=mu,
                    bld=self.bld[j],
                )
                transfer_function_3[i,j] = transfer_function(
                    omega=omega,
                    z=z,
                    cor_freq=cor_freq,
                    delta=self.delta[i],
                    mu=mu,
                    bld=self.bld[j]+delta_bld/2,
                )
                transfer_function_4[i,j] = transfer_function(
                    omega=omega,
                    z=z,
                    cor_freq=cor_freq,
                    delta=self.delta[i],
                    mu=mu,
                    bld=self.bld[j]-delta_bld/2,
                )
        self.assertTrue(
                np.shape(transfer_function_0) == (len(self.delta), len(self.bld))
            )    
        self.assertTrue(
                np.shape(transfer_function_1) == (len(self.delta), len(self.bld))
            )
        self.assertTrue(
                np.shape(transfer_function_2) == (len(self.delta), len(self.bld))
            )
        self.assertTrue(
                np.shape(transfer_function_3) == (len(self.delta), len(self.bld))
            )
        self.assertTrue(
                np.shape(transfer_function_4) == (len(self.delta), len(self.bld))
            )
