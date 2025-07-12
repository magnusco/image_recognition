import numpy as np
import pickle
import numpy.random as rnd
import matplotlib.pyplot as plt
import time


class NeuralNetwork:

    def __init__(self, M, h, epsilon, tau, TOL, running_time, beta, rho):
        self.M = M
        self.h = h
        self.eps = epsilon
        self.tau = tau
        self.TOL = TOL
        self.beta = beta
        self.rho = rho
        self.Y = None
        self.running_time = running_time
        self.W = np.ones(4)
        self.K = np.zeros((M, 4, 4))
        for i in range(0, self.M):
            self.K[i] = np.identity(4)

    # Learn to characterize points
    def learn(self, learning_points):
        Y0, C = self.set_data(learning_points)
        self.ngd(Y0, C)
        return 0

    # Test how many points the algorithm can characterize correctly
    def see(self, test_points):
        Y0, C = self.set_data(test_points)
        projection = np.around(self.eta(np.matmul(self.euler(Y0, self.K), self.W)))
        diff = np.abs(projection - C)
        correct = (1 - np.sum(diff) / len(diff)) * 100
        print("Characterized ", correct, " % of the points correctly.")
        return correct

    # Create data samples
    def make_circle_problem(self, n, nx, PLOT):
        # This python-script uses the following three input parameters:
        #   n       - Number of points.
        #   nx      - Resolution of the plotting.
        #   PLOT    - Boolean variable for plotting.

        # Defining function handles.
        transform_domain = lambda r: 2 * r - 1
        rad = lambda x1, x2: np.sqrt(x1 ** 2 + x2 ** 2)

        # Initializing essential parameters.
        r = np.linspace(0, 1, nx)
        x = transform_domain(r)
        dx = 2 / nx
        x1, x2 = np.meshgrid(x, x)

        # Creating the data structure 'problem' in terms of dictionaries.
        problem = {'domain': {'x1': x, 'x2': x}, 'classes': [None, None]}
        group1 = {'mean_rad': 0, 'sigma': 0.1, 'prob_unscaled': lambda x1, x2: 0, 'prob': lambda x1, x2: 0,
                  'density': 0}
        group1['prob_unscaled'] = lambda x, y: np.exp(
            -(rad(x, y) - group1['mean_rad']) ** 2 / (2 * group1['sigma'] ** 2))
        density_group1 = group1['prob_unscaled'](x1, x2)
        int_density_group1 = (dx ** 2) * sum(sum(density_group1))
        group1['density'] = density_group1 / int_density_group1
        group2 = {'mean_rad': 0.5, 'sigma': 0.1, 'prob_unscaled': lambda x1, x2: 0, 'prob': lambda x1, x2: 0,
                  'density': 0}
        group2['prob_unscaled'] = lambda x, y: np.exp(
            -(rad(x, y) - group2['mean_rad']) ** 2 / (2 * group2['sigma'] ** 2))
        density_group2 = group2['prob_unscaled'](x1, x2)
        int_density_group2 = (dx ** 2) * sum(sum(density_group2))
        group2['density'] = density_group2 / int_density_group2
        problem['classes'][0] = group1
        problem['classes'][1] = group2

        # Creating the arrays x1 and x2.
        x1 = np.zeros((n, 2))
        x2 = np.zeros((n, 2))
        count = 0
        for i in range(0, n):
            count += 1
            N1 = 'x1_' + str(count) + '.png'
            N2 = 'x2_' + str(count) + '.png'
            x1[i, 0], x1[i, 1] = self.pinky(problem['domain']['x1'], problem['domain']['x2'],
                                       problem['classes'][0]['density'], PLOT, N1)
            x2[i, 0], x2[i, 1] = self.pinky(problem['domain']['x1'], problem['domain']['x2'],
                                       problem['classes'][1]['density'], PLOT, N2)

        # Creating the data structure 'data' in terms of dictionaries.
        x = np.concatenate((x1[0:n, :], x2[0:n, :]), axis=0)
        y = np.concatenate((np.ones((n, 1)), 2 * np.ones((n, 1))), axis=0)
        i = rnd.permutation(2 * n)
        data = {'x': x[i, :], 'y': y[i]}

        return data, problem

    def pinky(self, Xin, Yin, dist_in, PLOT, NAME):
        # Checking the input.
        if len(np.shape(dist_in)) > 2:
            print("The input must be a N x M matrix.")
            return
        sy, sx = np.shape(dist_in)
        if (len(Xin) != sx) or (len(Yin) != sy):
            print("Dimensions of input vectors and input matrix must match.")
            return
        for i in range(0, sy):
            for j in range(0, sx):
                if dist_in[i, j] < 0:
                    print("All input probability values must be positive.")
                    return

        # Create column distribution. Pick random number.
        col_dist = np.sum(dist_in, 1)
        col_dist /= sum(col_dist)
        Xin2 = Xin
        Yin2 = Yin

        # Generate random value index and saving first value.
        ind1 = self.gendist(col_dist, 1, 1, PLOT, NAME)
        ind1 = np.array(ind1, dtype="int")
        x0 = Xin2[ind1]

        # Find corresponding indices and weights in the other dimension.
        A = (x0 - Xin) ** 2
        val_temp = np.sort(A)
        ind_temp = np.array([i[0] for i in sorted(enumerate(A), key=lambda x: x[1])])
        eps = 2 ** -52
        if val_temp[0] < eps:
            row_dist = dist_in[:, ind_temp[0]]
        else:
            low_val = min(ind_temp[0:2])
            high_val = max(ind_temp[0:2])
            Xlow = Xin[low_val]
            Xhigh = Xin[high_val]
            w1 = 1 - (x0 - Xlow) / (Xhigh - Xlow)
            w2 = 1 - (Xhigh - x0) / (Xhigh - Xlow)
            row_dist = w1 * dist_in[:, low_val] + w2 * dist_in[:, high_val]
        row_dist = row_dist / sum(row_dist)
        ind2 = self.gendist(row_dist, 1, 1, PLOT, NAME)
        y0 = Yin2[ind2]

        return x0, y0

    def gendist(self, P, N, M, PLOT, NAME):
        # Checking input.
        if min(P) < 0:
            print('All elements of first argument, P, must be positive.')
            return
        if (N < 1) or (M < 1):
            print('Output matrix dimensions must be greater than or equal to one.')
            return

        # Normalizing P and creating cumlative distribution.
        Pnorm = np.concatenate([[0], P], axis=0) / sum(P)
        Pcum = np.cumsum(Pnorm)

        # Creating random matrix.
        R = rnd.rand()

        # Calculate output matrix T.
        V = np.linspace(0, len(P) - 1, len(P))
        hist, inds = np.histogram(R, Pcum)
        hist = np.argmax(hist)
        T = int(V[hist])

        # Plotting graphs.
        if PLOT == True:
            Pfreq = (N * M * P) / sum(P)
            LP = len(P)
            fig, ax = plt.subplots()
            ax.hist(T, np.linspace(1, LP, LP))
            ax.plot(Pfreq, color='red')
            ax.set_xlabel('Frequency')
            ax.set_ylabel('P-vector Index')
            fig.savefig(NAME)

        return T

    # Creata new data samples
    def set_data(self, learning_points):
        data, problem = self.make_circle_problem(learning_points, 100, False)
        Y0 = np.concatenate((data['x'], data['x']**2), axis=1)
        C = np.ndarray.flatten(data['y']) - 1
        self.Y = np.zeros((self.M + 1, len(Y0), 4))
        return Y0, C

    # Function in euler method
    def sigma(self, x):
        return np.tanh(x)

    # Derivative of function in Euler method
    def sigma_der(self, x):
        return 1 / np.cosh(x)**2

    # Projection function
    def eta(self, x):
        return np.exp(x) / (np.exp(x) + 1)

    # Derivative of projection function
    def eta_der(self, x):
        return np.exp(x) / np.power(np.exp(x) + 1, 2)

    # Euler method, given by object constants
    def euler(self, Y0, K):
        Y = Y0
        self.Y[0] = Y0
        count = 1
        for i in range(0, self.M):
            Y = Y + self.h * self.sigma(np.matmul(Y, K[i]))
            self.Y[count] = Y
            count += 1
        return Y

    # Objective function, used to compare projected points to color data
    def obj_func(self, Y_M, W, C):
        x_vec = self.eta(np.matmul(Y_M, W)) - C
        J = 0.5 * np.dot(x_vec, x_vec)
        return J

    # Numerical calculation of the gradient of the objective function
    def numerical_grad_calc(self, Y0, C):
        Y_M = self.euler(Y0, self.K)
        dJ = np.zeros((self.M, 4, 4))
        dW = np.zeros(4)
        for m in range(0, self.M):
            for i in range(0, 4):
                for j in range(0, 4):
                    self.K[m][i][j] += self.eps
                    Y_M_alt = self.euler(Y0, self.K)
                    dJ[m][i][j] = (1 / self.eps) * (self.obj_func(Y_M_alt, self.W, C) - self.obj_func(Y_M, self.W, C))
                    self.K[m][i][j] -= self.eps
        for i in range(0, 4):
            self.W[i] += self.eps
            dW[i] = (1 / self.eps) * (self.obj_func(Y_M, self.W, C) - self.obj_func(Y_M, self.W, C))
            self.W[i] -= self.eps
        return dJ, dW

    # Analytical calculation of the gradient of the objective function
    def analytical_grad_calc(self, Y0, C):

        Y_M = self.euler(Y0, self.K)
        dJ_dW = np.matmul(np.transpose(Y_M), np.multiply(self.eta_der(np.matmul(Y_M, self.W)), self.eta(np.matmul(Y_M, self.W)) - C))
        dJ_dY= np.outer(np.multiply(self.eta_der(np.matmul(Y_M, self.W)), self.eta(np.matmul(Y_M, self.W)) - C), self.W)

        dJ_dK = np.zeros((self.M, 4, 4))
        U = dJ_dY
        for i in range(1, self.M):
            dJ_dK[self.M - i] = self.h*np.matmul(np.transpose(self.Y[self.M-i]),
                                            np.multiply(self.sigma_der(self.Y[self.M-i]@self.K[self.M-i]),U))
            U = U + self.h * np.matmul(np.multiply(self.sigma_der(np.matmul(self.Y[self.M - i], self.K[self.M - i])), U),
                                       np.transpose(self.K[self.M - i]))

        return dJ_dK, dJ_dW

    # Numerical gradient descent with constant learning rate
    def ngd_const_tau(self, Y0, C):
        count = 0
        start_time = time.time()
        Y_M = self.euler(Y0, self.K)
        res = self.obj_func(Y_M, self.W, C)
        print(res)
        while res > self.TOL:
            dJ, dW = self.numerical_grad_calc(Y0, C)
            self.K -= self.tau * dJ
            self.W -= self.tau * dW
            Y_M = self.euler(Y0, self.K)
            res = self.obj_func(Y_M, self.W, C)
            print(res)
            count += 1
        print("Iterations: ", count)
        print("Time: ", time.time() - start_time)
        return 0

    # Numerical gradient descent with variable learning rate
    def ngd(self, Y0, C):
        start_time = time.time()
        min_res = self.obj_func(self.euler(Y0, self.K), self.W, C)
        count = 0
        while True:
            tau, save_tau = self.tau, self.tau
            dJ, dW = self.analytical_grad_calc(Y0, C)
            last_res = min_res
            for i in range(0, 10):
                tau *= self.rho
                res = self.obj_func(self.euler(Y0, self.K - tau * dJ), self.W - tau * dW, C)
                if res < min_res:
                    min_res = res
                    save_tau = tau
                if res <= self.beta * last_res:
                    break
            self.K -= save_tau * dJ
            self.W -= save_tau * dW
            print(min_res)
            count += 1
            if min_res < self.TOL or time.time() - start_time > self.running_time:
                print("Iterations: ", count)
                print("Time: ", time.time() - start_time)
                break
        return 0

    # Write learned parameters to file in binary
    def write_to_file(self):
        with open('weights.pkl', 'wb') as f:
            pickle.dump([self.K, self.W], f)
        return 0

    # Read parameters from file, save in the object
    def read_from_file(self):
        with open('weights.pkl', 'rb') as f:
            self.K, self.W = pickle.load(f)
        return 0

    def plot_decision_boundary(self, test_points, filename="media/decision_boundary.png"):
         
        Y0, C = self.set_data(test_points)
        
        x_min, x_max = Y0[:, 0].min() - 0.1, Y0[:, 0].max() + 0.1
        y_min, y_max = Y0[:, 1].min() - 0.1, Y0[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_features = np.concatenate((grid_points, grid_points**2), axis=1)
        

        self.Y = np.zeros((self.M + 1, len(grid_features), 4))

        Z = self.eta(np.matmul(self.euler(grid_features, self.K), self.W))
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(8, 8))
        plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.5)
        plt.scatter(Y0[C==0, 0], Y0[C==0, 1], c='red', edgecolors='k', label='Class 0')
        plt.scatter(Y0[C==1, 0], Y0[C==1, 1], c='blue', edgecolors='k', label='Class 1')
        plt.title("Learned Decision Boundary")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(filename)
        print(f"Decision boundary plot saved to {filename}")