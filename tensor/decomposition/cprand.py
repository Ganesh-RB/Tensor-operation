import numpy as np
import time
import matplotlib.pyplot as plt
from tensor.operation.kruskal import kruskal
from tensor.operation.matricize import matricize
from tensor.operation.sampled_khatri_rao_prod import SKR
from tensor.operation.sampled_matricize import Sampled_Matricize
from tensor.operation.stopping_criteria import StoppingCriteria


class CP_RAND:
    def __init__(self, tensor: np.ndarray, rank: int, max_iter=10000, eps=0.01, initalisation: str = "random"):
        """
        PARAFAC decompositon, it initializes the maximum amount of iterations that 
        are done along with the tolerance value for the convergence
        it also takes the size of the core tensor `rank`
        """
        # Original Tensor
        self.tensor = tensor

        # Rank of decomposition being targeted
        self.rank = rank

        # Max iterations
        self.max_iter = max_iter

        # Stopping error value
        self.eps = eps

        # sets the type of initialisation
        self.init_type = initalisation
        # Stores the errrors at each step
        self.errors = []
        # Factor matrixes
        self.factors = []

        # Current lamda value
        self.lamda = np.ones(self.rank)

        # statistics measures
        self.statistics = {
            "iterations": [],
            "errors": [],
            "fit": [],
            "cum_fit_time": [],
            "init_time": 0,
        }


    def fit(self, check_convergence=True, stopping_criteria=None):
        # training loop,self explanatory
        self._init_factors()
        prev_fitting_time = 0
        start = time.time()
        P = np.power(2, 14)
        for i in range(self.max_iter):
            self.statistics["iterations"].append(i)
            for mode in (range(self.tensor.ndim)):
                self._update_factors(mode)

            if check_convergence and stopping_criteria == "sampled":
                if StoppingCriteria(self.tensor, kruskal(*self.factors), P, threshold=self.eps):
                    break

            elif check_convergence and self._is_converged():
                break

            self.statistics["cum_fit_time"].append(prev_fitting_time + time.time()-start)
            prev_fitting_time = self.statistics["cum_fit_time"][-1]
            start = time.time()

 
        if check_convergence == False:
            self.statistics["errors"].append(np.linalg.norm(self.tensor - kruskal(*self.factors)))
            self.statistics["fit"].append(1 - (self.statistics["errors"][-1] / np.linalg.norm(self.tensor)))



    def _init_factors(self):
        """
        initialize the factors using the `rank` many left singular 
        vectors of the corresponding mode-n matricization of the input tensor
        """
        start = time.time()
        if self.init_type == 'random':
            self.factors = [np.random.rand(
                self.tensor.shape[i], self.rank) for i in range(self.tensor.ndim)]
        elif self.init_type == 'hosvd':
            self.factors = []
            for i in range(self.tensor.ndim):
                M = np.linalg.svd(matricize(self.tensor, i))[0]
                if M.shape[1] < self.rank:
                    M_ = np.zeros((M.shape[0], self.rank - M.shape[1]))
                    M = np.concatenate((M, M_), axis=1)
                else:
                    M = M[:, :self.rank]
                self.factors.append(M)
        else:
            raise Exception("Invalid initialisation method")
        self.statistics["init_time"] = time.time() - start


    def _update_factors(self, mode):
        """
        Update the factors per iteration for the `mode`'th Factor
        """
        tot_row = 1
        for x in range(self.tensor.ndim):
            if x != mode:
                tot_row *= self.tensor.shape[x]
        S = 10*self.rank
        # Very weird behaviour for this one
        sel_rows = np.random.choice(tot_row, S, replace=True)

        # This converges quite well
        # sel_rows=np.random.permutation(tot_row)

        Z_s = SKR(sel_rows, self.factors, mode)
        X_s = Sampled_Matricize(sel_rows, self.tensor, mode)
        # print(X_s.shape)
        # print(Z_s.shape)
        # print(len(sel_rows))
        # print(self.rank)
        self.factors[mode] = X_s @ np.linalg.pinv(Z_s.T)
        # print(self.factors[mode].shape)

        col_norms = np.linalg.norm(self.factors[mode], axis=0)

        self.factors[mode] = self.factors[mode] / \
            col_norms  # normalize the self.factors[mode]

        self.lamda = col_norms

    def _get_error(self):
        """
        compute the error between the current tensor
        and the tensor obtained by multiplying the factors
        """
        tmp = self.factors[0]
        norm = np.linalg.norm(self.tensor)

        self.factors[0] = self.factors[0] * self.lamda
        estim = kruskal(*self.factors)
        error = np.linalg.norm(self.tensor - estim)
        self.factors[0] = tmp

        return error

    def _is_converged(self):
        """
        check if the algorithm has converged
        by computing the Frobenius norm of the difference between the current tensor
        and the tensor obtained by multiplying the factors
        """
        tmp = self.factors[0]
        norm = np.linalg.norm(self.tensor)

        self.factors[0] = self.factors[0] * self.lamda
        estim = kruskal(*self.factors)
        error = np.linalg.norm(self.tensor - estim)
        # print("Error = ", error)
        self.statistics["errors"].append(error)
        self.statistics["fit"].append(1 - error / np.linalg.norm(self.tensor))
        # print("Error = ", error, "Fit = ", self.statistics["fit"][-1])
        self.errors.append(error)
        self.factors[0] = tmp
        return (error/norm) < self.eps
    

    def plot_errors(self):
        plt.plot(self.errors)
        plt.legend(["Errors"])
        plt.xlabel("Iterations")
        plt.ylabel("Frobenius norm")
        plt.show()

    def decompose(X, rank, max_iter=10000, eps=0.01, init_type='random', check_convergence=True, stoppig_criteria = None,  fit = True):
        """
        Decompose the tensor `X` into a sum of `rank` many rank-1 tensors
        """
        cp = CP_RAND(X, rank, max_iter, eps, init_type)
        if fit:
            cp.fit(check_convergence, stoppig_criteria)
        return cp.factors, cp.statistics