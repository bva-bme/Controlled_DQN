import random
import torch
import numpy as np
import scipy.linalg

# Hinf controller parameters for Cartpole (with dimensionality reduction)
#controller_A = torch.tensor([[-15188.3761787050, 5246.40717608589, 1319.88465425364], [-5002.91529467321, -3.85976063010519, -20.9948078012109], [-1309.51789398375, 3.12388744780353, -894.716940267115]], dtype=torch.float64)
#controller_B = torch.tensor([[0.0673986499973741, -81.9707419513076, -83.6735779388722], [-0.138909146087354, 1.30661821589823, -1.26499583822529], [-0.270344600507911, -3.83694496768237, -3.09536317442650]], dtype=torch.float64)
#controller_C = torch.tensor([-117.134269630233, 1.82394108027175, 4.93724737221972], dtype=torch.float64)

controller_A = torch.tensor([[-20470, 13550], [-13130, -2.394]], dtype=torch.float64)
controller_B = torch.tensor([[155.7, 53.97, 55.91], [-1.744, -0.6808, 0.179]], dtype=torch.float64)
controller_C = torch.tensor([174.1, -1.881], dtype=torch.float64)

# Define the learning agent
class DQNAgent(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, lr=1e-4, eps=1, eps_dec=0.001, gamma=0.9, 
                 environment='WrongEnv'):
        super(DQNAgent, self).__init__()

        self.learning_rate = lr
        self.epsilon = eps
        self.epsilon_decay = eps_dec
        self.discount_factor = gamma

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.hidden_layer = torch.nn.Linear(self.input_size, self.hidden_size, bias=True).double()
        self.output_layer = torch.nn.Linear(self.hidden_size, self.output_size, bias=True).double()

        self.neural_tangent_kernel = None
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        self.ctr = 0
        self.r_int = 0
        self.Q1_int = 0
        self.Q2_int = 0
        self.controller_internal_states = torch.zeros([controller_C.shape[0]], dtype=torch.float64)
        self.xi = 0

        self.environment = environment

        self.loss_arr = []
        self.control_input_arr = []

    def forward(self, x):
        x = torch.from_numpy(x)
        x = self.hidden_layer(x)  # b*x
        x = torch.relu(x)  # sigma(b*x)
        x = self.output_layer(x)   # y = a*sigma(b*x) + c
        return x

    # Compute the gradients
    def __partial_derivatives(self, x):
        self.zero_grad()

        w1 = torch.empty(self.output_size, self.hidden_size * self.input_size + self.hidden_size, dtype=torch.float64)
        w2 = torch.empty(self.output_size, self.hidden_size * self.output_size + self.output_size, dtype=torch.float64)

        x = x.double()
        x = x.numpy()

        for i in range(self.output_size):

            y = self.forward(x)
            y = y[0][i]
            y.backward()  # handle multiple outputs

            wi1 = self.hidden_layer.weight.grad  # nabla_theta f(x) (Jacobian)
            wi1 = torch.reshape(wi1, [wi1.shape[0] * wi1.shape[1], 1])
            wi1 = torch.cat([wi1, self.hidden_layer.bias.grad.unsqueeze(1)])

            wi2 = self.output_layer.weight.grad
            wi2 = torch.reshape(wi2, [wi2.shape[0] * wi2.shape[1], 1])
            wi2 = torch.cat([wi2, self.output_layer.bias.grad.unsqueeze(1)])

            wi1g = wi1.clone().detach()
            wi2g = wi2.clone().detach()  # create deep copy, otherwise gradients keep rolling

            w1[i] = wi1g.squeeze()
            w2[i] = wi2g.squeeze()

            self.zero_grad()

        return w1, w2

    # Compute the NTK
    def compute_neural_tangent_kernel(self, x):
        kernel = torch.zeros([x.shape[0] * self.output_size, x.shape[0] * self.output_size], dtype=torch.float64, 
                             requires_grad=False)
        i = 0
        for x1 in x.data:
            w1x1, w2x1 = self.__partial_derivatives(x1.unsqueeze(dim=0))
            j = 0
            for x2 in x.data:
                # sum_{i=1}^m  (df(x1)/dw1)^T*(df(x2)/dw1) + ...
                w1x2, w2x2 = self.__partial_derivatives(x2.unsqueeze(dim=0))
                kernel[self.output_size * i:self.output_size * i + self.output_size, 
                       self.output_size * j:self.output_size * j + self.output_size] \
                    = torch.matmul(w1x1, w1x2.transpose(0, 1)) + torch.matmul(w2x1, w2x2.transpose(0, 1))  
                j += 1
            i += 1
        self.neural_tangent_kernel = kernel
        return kernel

    # Epsilon greedy action selection
    def select_action(self, state):
        # Epsilon greedy
        eps = random.uniform(0, 1)
        if eps > self.epsilon:
            action = torch.argmax(self.forward(state)).item()
        else:
            action = random.randint(0, self.output_size - 1)
            if self.epsilon >= 0.01:
                self.epsilon -= self.epsilon_decay  # Leave some exploration
        return action

    # Train with SGD and quadratic loss (uncontroller)
    def train_network(self, state, action, reward, state_next):

        q_act = self.forward(state)[action]
        with torch.no_grad():
            q_next = torch.max(self.forward(state_next))

        # Compute the expected Q values
        expected_state_action_values = q_next * self.discount_factor + reward

        # Optimize the model
        loss = 0.5 * (expected_state_action_values - q_act)**2
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    # Compute Q1 change
    def predict_training(self, state, action, reward, state_next):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        neural_tangent_kernel = self.compute_neural_tangent_kernel(state_tensor)

        q_act = self.forward(state)[action]
        q_next = torch.max(self.forward(state_next))

        # Compute the expected Q values
        expected_state_action_values = q_next * self.discount_factor + reward
        dq1 = self.learning_rate * neural_tangent_kernel[action, action] * (expected_state_action_values - q_act)

        return dq1

    # Compute Q2 change
    def predict_other_state(self, state_original, state_other, action, reward, state_next):
        state_original_tensor = torch.from_numpy(state_original).float().unsqueeze(0)
        state_other_tensor = torch.from_numpy(state_other).float().unsqueeze(0)
        state_tensor = torch.cat([state_original_tensor, state_other_tensor])
        neural_tangent_kernel = self.compute_neural_tangent_kernel(state_tensor)

        q_act = self.forward(state_original)[action]
        q_next = torch.max(self.forward(state_next))

        # Compute the expected Q values
        expected_state_action_values = q_next * self.discount_factor + reward
        dq2 = self.learning_rate * neural_tangent_kernel[action, self.output_size + action] * \
            (expected_state_action_values - q_act)

        return dq2, neural_tangent_kernel

    # Compute the state space and perform pole placement to stabilize the learning
    def compute_temporal_difference_state_space(self, state, action, reward, state_next):

        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        state_next_tensor = torch.from_numpy(state_next).float().unsqueeze(0)

        state_cat_tensor = torch.cat([state_tensor, state_next_tensor])

        q1 = self.forward(state)[action]
        q2 = torch.max(self.forward(state_next))
        a2 = torch.argmax(self.forward(state_next)).item()

        Theta1 = self.compute_neural_tangent_kernel(state_tensor)[action, action]
        Theta2 = self.compute_neural_tangent_kernel(state_cat_tensor)[self.output_size + a2, a2]

        A = torch.tensor([[-Theta1, self.discount_factor * Theta1], [-Theta2, self.discount_factor * Theta2]])
        Bw = torch.tensor([Theta1, Theta2])
        Bu = torch.tensor([Theta1, 0])

        x = torch.tensor([q1, q2])
        w = torch.tensor([reward], dtype=torch.float64)

        # Sanity check
        sanity_check = False
        if sanity_check:
            dq1 = self.predict_training(state, action, reward, state_next)
            dq2, tmp = self.predict_other_state(state, state_next, action, reward, state_next)
            dq = self.learning_rate * (torch.matmul(A, x) + Bw * w)
            print("Done! Errors: ", (dq[0] - dq1).item(), (dq[1] - dq2).item())

        # Eigenvalues of A
        eig = torch.linalg.eigvals(A)
        # print("Eigenvalues: ", eig)

        # New poles
        p = torch.zeros([2])
        if eig[0].real > 0:
            p[0] = -0.001
        else:
            p[0] = eig[0].real
        if eig[1].real > 0:
            p[1] = -0.001
        else:
            p[1] = eig[1].real

        # Do the pole placement
        k = self.__pole_placement(A, Bu, p)

        return Theta1, Theta2, k[0], k[1]

    # LQ control with the augmented state space
    def __lq_servo(self, state, action, state_next):

        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        state_next_tensor = torch.from_numpy(state_next).float().unsqueeze(0)

        state_cat_tensor = torch.cat([state_tensor, state_next_tensor])

        a2 = torch.argmax(self.forward(state)).item()

        Theta1 = self.compute_neural_tangent_kernel(state_tensor)[action, action]
        Theta2 = self.compute_neural_tangent_kernel(state_cat_tensor)[self.output_size + a2, a2]

        A_aug = torch.tensor([[-Theta1, self.discount_factor * Theta1, 0], [-Theta2, self.discount_factor * Theta2, 0],
                              [-1, 0, 0]])
        B_aug = torch.tensor([Theta1, 0, 0])

        C, st_ctrb = self.__check_controllability(A_aug, B_aug)
        if not st_ctrb:
            return Theta1, Theta2, 0, 0, 0

        # Controller weights
        if self.environment == 'CartPole':
            Q = torch.tensor([[1, 0, 0], [0, 30000, 0], [0, 0, 10]])
            R = torch.tensor([20000])
        elif self.environment == 'Acrobot':
            Q = torch.tensor([[1, 0, 0], [0, 100, 0], [0, 0, 80]])
            R = torch.tensor([30])
        elif self.environment == 'MountainCar':
            Q = torch.tensor([[1, 0, 0], [0, 10, 0], [0, 0, 100]])
            R = torch.tensor([1000])
        elif self.environment == 'LunarLander':
            Q = torch.tensor([[0, 0, 0], [0, 20000, 0], [0, 0, 100]])
            R = torch.tensor([10000])
        else:
            print('Wrong environment!')
            exit()

        k = - self.__lqr(A_aug, B_aug, Q, R)

        return Theta1, Theta2, k[2], k[1], k[0]

    @staticmethod
    def __lqr(A, B, Q, R):
        # first, try to solve the ricatti equation
        A = A.numpy()
        B = B.unsqueeze(0)
        B = B.numpy().T
        Q = Q.numpy()
        R = R.unsqueeze(0)
        R = R.numpy()

        try:
            X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
        except:
            return torch.zeros([3])

        # compute the LQR gain
        K = np.matrix(scipy.linalg.inv(R) * (B.T * X))
        eigVals, eigVecs = scipy.linalg.eig(A - B * K)
        K = torch.from_numpy(K)[0]

        return K

    @staticmethod
    def __check_controllability(A, b):
        c1 = b.unsqueeze(1)
        c2 = torch.matmul(A, b).unsqueeze(1)
        C = torch.hstack((c1, c2))
        r_C = torch.linalg.matrix_rank(C)
        if r_C == 2:
            st_ctrb = True  # print("The system is controllable! :)")
        else:
            print("The system is NOT controllable! :(")
            st_ctrb = False
        return C, st_ctrb

    def __pole_placement(self, A, b, p):
        eig = torch.linalg.eigvals(A)

        a1 = -eig[0].real - eig[1].real
        a0 = (-eig[0].real)*(-eig[1].real)

        a1_bar = -p[0] - p[1]
        a0_bar = (-p[0])*(-p[1])

        k1 = a1_bar - a1
        k0 = a0_bar - a0
        kc = torch.tensor([k0, k1])

        C, st_ctrb = self.__check_controllability(A, b)
        if st_ctrb:
            Tau = torch.tensor([[1, a1], [0, 1]])
            T = torch.inverse(torch.matmul(C, Tau))
            k = torch.matmul(T, kc)
        else:
            k = torch.tensor([0.0, 0.0], dtype=torch.float64)

        return k

    # Learning with pole placement
    def poleplacement_controlled_learning(self, state, action, reward, state_next):

        q_act = self.forward(state)[action]
        with torch.no_grad():
            q_next = torch.max(self.forward(state_next))

        Theta1, Theta2, k1, k2 = self.compute_temporal_difference_state_space(state, action, reward, state_next)

        loss = 1/(2*(1 - Theta1*k1)) * (reward + (self.discount_factor + Theta1 * k2) *q_next - (1 - Theta1 * k1) * q_act)**2

        # Optimize the model
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    # Learning with LQ control
    def lq_controlled_learning(self, state, action, reward, state_next, done):
        q_act = self.forward(state)[action]
        Theta1, Theta2, k1, k2, k3 = self.__lq_servo(state, action, state_next)
        k3 = -k3

        with torch.no_grad():
            q_next = torch.max(self.forward(state_next))

            self.r_int += reward * self.learning_rate
            self.Q1_int += q_act * self.learning_rate
            self.Q2_int += q_next * self.learning_rate

            self.xi = self.r_int + self.discount_factor * self.Q2_int - self.Q1_int

            u = - k2 * q_next - k1 * q_act + k3 * self.xi

        # print(k1, k2, k3)
        loss = 1 / (2 * (1 + k1)) * (reward + (self.discount_factor - k2) * q_next -
                                     (1 + k1) * q_act + k3 * self.xi) ** 2

        if torch.isnan(loss):
            print('The loss is NaN! ERROR. Quitting.')
            exit()

        self.loss_arr.append(loss.item())
        self.control_input_arr.append(u.item())

        # Optimize the model
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Reset internal states
        if done:
            self.ctr = self.ctr + 1
            self.xi = 0
            if self.ctr == 1:
                self.r_int = 0
                self.Q1_int = 0
                self.Q2_int = 0
                self.ctr = 0

    # Fixed structure Hinf controlled learning
    def robust_hinf_learning(self, state, action, reward, state_next, done):

        # Fixed structure hinf controller parameters - from MATLAB (Robust_Hinf.m)
        if self.environment == 'CartPole':
            k1 = -0.379
            k2 = 2.82
            k3 = 0.1937
            ctr_thd = 300000
        elif self.environment == 'Acrobot':
            k1 = -0.00174
            k2 = 0.39
            k3 = 3.115
            ctr_thd = 1
        elif self.environment == 'MountainCar':
            k1 = -0.00379
            k2 = 0.0382
            k3 = 100.0115
            ctr_thd = 1
        elif self.environment == 'LunarLander':
            k1 = -0.00174
            k2 = 0.39
            k3 = 22.115
            ctr_thd = 1
        else:
            print('Wrong environment!')
            exit()

        q_act = self.forward(state)[action]
        with torch.no_grad():
            q_next = torch.max(self.forward(state_next))

            self.r_int += reward * self.learning_rate
            self.Q1_int += q_act * self.learning_rate
            self.Q2_int += q_next * self.learning_rate

            self.xi = self.r_int + self.discount_factor * self.Q2_int - self.Q1_int

            u = - k2 * q_next - k1 * q_act + k3 * self.xi

        loss = 1 / (2 * (1 + k1)) * (reward + (self.discount_factor - k2) * q_next -
                                     (1 + k1) * q_act + k3 * self.xi) ** 2

        if torch.isnan(loss):
            print(self.xi, q_act, q_next)
            print('The loss is NaN! ERROR. Quitting.')
            exit()

        self.loss_arr.append(loss.item())
        self.control_input_arr.append(u.item())

        # Optimize the model
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        # Reset internal states
        if done:
            self.ctr = self.ctr+1
            self.xi = 0
            if self.ctr == ctr_thd:
                self.r_int = 0
                self.Q1_int = 0
                self.Q2_int = 0
                self.ctr = 0

    # Hinf controlled learning
    def robust_dynamic_hinf_learning(self, state, action, reward, state_next, done=False):

        q_act = self.forward(state)[action]
        with torch.no_grad():
            q_next = torch.max(self.forward(state_next))

            self.r_int += reward * self.learning_rate
            self.Q1_int += q_act * self.learning_rate
            self.Q2_int += q_next * self.learning_rate

            self.xi = self.r_int + self.discount_factor * self.Q2_int - self.Q1_int

            u_c = torch.tensor([self.xi, q_act, q_next])
            self.controller_internal_states += \
                self.learning_rate*(torch.mv(controller_A, self.controller_internal_states)+torch.mv(controller_B, u_c))
            u = torch.dot(controller_C, self.controller_internal_states)
        loss = 0.5 * (reward + self.discount_factor * q_next - q_act - u) ** 2

        self.loss_arr.append(loss.item())
        self.control_input_arr.append(u.item())

        if done:
            self.xi = 0
            self.controller_internal_states = torch.zeros([controller_C.shape[0]], dtype=torch.float64)

        if torch.isnan(loss):
            print('The loss is NaN! ERROR. Quitting.')
            exit()

        # Optimize the model
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
