import matplotlib.pyplot as plt
from neural_net import NeuralNetwork


if __name__ == '__main__':
    # Number of steps in euler
    M = 6
    # Step-size
    h = 0.4
    # Numerical differentiation factor
    epsilon = 0.0001
    # Learning rate
    tau = 2
    # Tolerance for objective function
    TOL = 1
    # Shrinking factor
    rho = 0.5
    # Satisfactory descent rate
    beta = 0.95
    # Time allowed for the algorithm to run (seconds)
    running_time = 1.1

    # x = []
    # correct = []
    # for i in range(0, 10):
    #     net = NeuralNetwork(M, h, epsilon, tau, TOL, running_time, beta, rho)
    #     net.learn(500)
    #     y = net.see(1000)
    #     x.append(i)
    #     correct.append(y)

    # plt.plot(x, correct, 'k')
    # plt.plot([0, 9], [98, 98], 'r', label=r"98 % correct")
    # plt.plot([0, 9], [99, 99], 'b', label=r"99 % correct")
    # plt.legend(fontsize=13)
    # plt.title("Correct points after 1.1 second", fontsize=15)
    # plt.xlabel("Try nr.")
    # plt.show()


    print("\nTraining a final model for visualization...")
    final_net = NeuralNetwork(M, h, epsilon, tau, TOL, 5, beta, rho) # Train for 5 seconds
    final_net.learn(500)
    final_net.plot_decision_boundary(200, filename="media/decision_boundary.png")