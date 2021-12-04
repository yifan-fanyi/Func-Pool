import numpy as np

class HMM:
    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L
        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S, L = len(self.pi), len(Osequence)
        alpha = np.zeros([S, L])
        alpha[:,0] = self.pi * self.B[:,self.obs_dict[Osequence[0]]]
        for t in range(1, L):
            for s in range(S):
                alpha[s,t] = self.B[s,self.obs_dict[Osequence[t]]] * np.dot(self.A[:,s], alpha[:,t-1])
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L
        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S, L = len(self.pi), len(Osequence)
        beta = np.zeros([S, L])
        beta[:, L-1] = 1
        for t in reversed(range(L - 1)):
            for i in range(S):
                beta[i, t] = sum([beta[j, t + 1] * self.A[i, j] * self.B[j, self.obs_dict[Osequence[t + 1]]] for j in range(S)])
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L
        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        alpha = self.forward(Osequence)
        for i in range(len(self.pi)):
            prob += alpha[i, len(Osequence) - 1]
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L
        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S, L = len(self.pi), len(Osequence)
        prob = np.zeros([S, L])
        beta = self.backward(Osequence)
        alpha = self.forward(Osequence)
        a = 0
        for s in range(S):
            a += alpha[s, len(Osequence) - 1]
        for i in range(len(Osequence)):
            for s in range(S):
                b = alpha[s, i] * beta[s, i]
                prob[s][i] = b / a
        return prob

    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L
        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S, L = len(self.pi), len(Osequence)
        prob = np.zeros([S, S, L - 1])
        beta = self.backward(Osequence)
        alpha = self.forward(Osequence)
        a = 0
        for s in range(S):
            a += alpha[s][L - 1]
        for i in range(L - 1):
            for s in range(S):
                for ss in range(S):
                    prob[s, ss, i] = self.A[s, ss] * \
                                     self.B[ss, self.obs_dict[Osequence[i + 1]]] * \
                                     beta[ss, i + 1] * alpha[s, i] / a
        return prob

        
    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L
        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        S, L = len(self.pi), len(Osequence)
        delta_b = np.zeros([S, L]).astype('int16')
        delta = np.zeros([S, L])
        delta[:, 0] = self.pi * self.B[:, self.obs_dict[Osequence[0]]]
        for t in range(1, L):
            for i in range(S):
                delta[i, t] = self.B[i, self.obs_dict[Osequence[t]]] * np.max(self.A[:, i] * delta[:, t-1])
                delta_b[i, t] = np.argmax(self.A[:, i] * delta[:, t-1])
        z = np.argmax(delta[:, L - 1])
        path.append(z)
        for t in range(L - 1, 0, -1):
            z = delta_b[(int)(z),t]
            path.append(z)
        path = path[::-1]
        states = [0] * len(path)
        for i in self.state_dict:
            for j in range(len(path)):
                if path[j] == self.state_dict[i]:
                    states[j]=i
        return states

    