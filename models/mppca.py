# packages
import torch
from typing import Tuple


class MPPCA():
    """
    Initialize a probabilistic model representing a Mixture of Probabilistic 
    Principal Component Analyzers (MPPCA) [1]. This model constrains the 
    covariance matrices of a Gaussian mixture model to be low-rank and diagonal.
    
    Original publication:
    [1] Tipping, M. E., & Bishop, C. M. (1999). Mixtures of Probabilistic 
        Principal Component Analyzers. Neural Computation, 11(2), 443-482.

    The implementation is based on the open-source code from
    [2] Richardson, E., & Weiss, Y. (2018). On GANs and GMMs. Advances in 
        Neural Information Processing Systems, 31.
        
    Args:
        n_components (int): number of mixture components (alias: k).
        n_features (int): number of input dimensions (alias: d).
        n_factors (int): number of underlying factors (alias: l).
    
    Learnable Parameters:
        mu (torch.Tensor): (k, d) tensor of component mean vectors.
        W (torch.Tensor): (k, d, l) tensor of factor loading matrices.
        log_Psi (torch.Tensor): (k, d) tensor of log diagonal noise values.
        pi_logits (torch.Tensor): (k,) tensor of mixing proportion logits.
    """

    def __init__(self, n_components, n_features, n_factors):
        super(MPPCA, self).__init__()
        self.n_components = n_components
        self.n_features = n_features
        self.n_factors = n_factors

        self.mu = torch.zeros(n_components, n_features)
        self.W = torch.zeros(n_components, n_features, n_factors)
        self.log_Psi = torch.zeros(n_components, n_features)
        self.pi_logits = torch.log(torch.ones(n_components)/float(n_components))

    def sample(self, n:int) -> torch.Tensor:
        """
        Sample from the learned generative model.

        Args:
            n (int): total number of samples to draw.

        Returns:
            samples (torch.Tensor): (n, d) tensor of generated samples.
        """

        k, d, l = self.W.shape
        # sample mixture components
        def clip_pi(pi, min_weight=1e-3):
            # Ensure no weight is below the minimum
            clipped_pi = torch.clamp(pi, min=min_weight)
            # Renormalize the weights to sum to 1
            normalized_pi = clipped_pi / clipped_pi.sum(dim=-1, keepdim=True)
            return normalized_pi
    
        pi = clip_pi(torch.softmax(self.pi_logits, dim=0))
        components = torch.multinomial(pi, num_samples=n, replacement=True)
        z_l = torch.randn(n, l) # latent variables
        z_d = torch.randn(n, d) # noise 

        Wz = self.W[components] @ z_l[..., None]
        mu = self.mu[components][..., None]
        epsilon = (z_d * torch.exp(0.5*self.log_Psi[components]))[..., None]
        
        samples = (Wz + mu + epsilon).squeeze()

        return samples
    
    def component_log_prob(self, x:torch.Tensor) -> torch.Tensor:
        """
        Compute the log-likelihoods associated with each mixture component.

        Args:
            x (torch.Tensor): [N x D] tensor of input data

        Returns:
            component_log_probs (torch.Tensor): [N x K] tensor of per-component 
            log-likelihoods 
        """

        mu = self.mu
        W = self.W
        log_Psi = self.log_Psi
        pi_logits = self.pi_logits

        k, d, l = W.shape
        WT = W.transpose(1,2)
        inv_Psi = torch.exp(-log_Psi).view(k,d,1)
        I = torch.eye(l, device=W.device).reshape(1,l,l)
        L = I + WT @ (inv_Psi * W)
        inv_L = torch.linalg.solve(L, I)

        # compute Mahalanobis distance using the matrix inversion lemma
        def mahalanobis_distance(i):
            x_c = (x - mu[i].reshape(1,d)).T
            component_m_d = (inv_Psi[i] * x_c) - \
                ((inv_Psi[i] * W[i]) @ inv_L[i]) @ (WT[i] @ (inv_Psi[i] * x_c))
            
            return torch.sum(x_c * component_m_d, dim=0)

        # combine likelihood terms
        m_d = torch.stack([mahalanobis_distance(i) for i in range(k)])
        log_det_cov = torch.logdet(L) - \
            torch.sum(torch.log(inv_Psi.reshape(k,d)), axis=1)
        log_const = torch.log(torch.tensor(2.0)*torch.pi)
        log_probs = -0.5 * ((d*log_const + log_det_cov).reshape(k, 1) + m_d)

        component_log_probs = pi_logits.reshape(1,k) + log_probs.T
        
        return component_log_probs

    def log_prob(self, x:torch.Tensor) -> torch.Tensor:
        """
        Computes the per-sample log-likelihoods.

        Args:
            x (torch.Tensor): (n, d) tensor of input data.

        Returns:
            log_probs (torch.Tensor): (n,) tensor of per-sample log-likelihoods. 
        """

        log_probs = torch.logsumexp(self.component_log_prob(x), dim=1)

        return log_probs
    
    def expectation(self, x:torch.Tensor, weights:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the expected log-likelihood given the current parameters.

        Args:
            x (torch.Tensor): (n, d) tensor of input data.
            weights (torch.Tensor): (n,) tensor of importance weights.

        Returns:
            r (torch.Tensor): (n, k) tensor of per-component responsibilities.
            llh (torch.Tensor): total log-likelihood of the input data.
        """

        component_log_probs = self.component_log_prob(x)
        log_probs = self.log_prob(x).reshape(-1, 1)

        r = torch.exp(component_log_probs - log_probs)
        llh = torch.sum(weights * log_probs) / torch.sum(weights)
        
        return [r, llh]
    
    def maximization(self, x:torch.Tensor, weights:torch.Tensor, r:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the parameters that maximize the expected log-likelihood. All 
        equations and appendices reference Tipping and Bishop (1999) [1].

        Args:
            x (torch.Tensor): (n, d) tensor of input data.
            weights (torch.Tensor): (n,) tensor of importance weights.
            r (torch.Tensor): (n, k) tensor of per-component responsibilities.

        Returns:
            mu (torch.Tensor): (k, d) tensor of component mean vectors.
            W (torch.Tensor): (k, d, l) tensor of factor loading matrices.
            log_Psi (torch.Tensor): (k, d) tensor of log diagonal noise values.
            pi_logits (torch.Tensor): (k,) tensor of mixing proportion logits.
        """

        k, d, l = self.W.shape
        n = x.shape[0]
        r = weights * r
        r_sum = torch.sum(r, dim=0)
        if any(r_sum < 1e-6): # prevent division by zero
            r_sum += 1e-6

        def per_component_m_step(i):
            # (C.8)
            mui_new = torch.sum(r[:, [i]] * x, dim=0) / r_sum[i]
            sigma2_I = torch.exp(self.log_Psi[i, 0]) * torch.eye(l)
            inv_Mi = torch.inverse(self.W[i].T @ self.W[i] + sigma2_I + 1e-8*torch.eye(self.n_factors))
            x_c = x - mui_new.reshape(1, d)
            # efficiently calculate (Si)(Wi) as discussed in Appendix C
            SiWi = (1.0/r_sum[i]) * (r[:, [i]]*x_c).T @ (x_c @ self.W[i])
            # (C.14)
            Wi_new = SiWi @ torch.inverse(sigma2_I + inv_Mi @ self.W[i].T @ SiWi)
            # (C.15)
            t1 = torch.trace(Wi_new.T @ (SiWi @ inv_Mi))
            trace_Si = torch.sum(n/r_sum[i] * torch.mean(r[:, [i]]*x_c*x_c, dim=0))
            sigma_2_new = (trace_Si - t1)/d
            if sigma_2_new < 1e-6: # prevent taking log of small numbers
                sigma_2_new += 1e-6

            return mui_new, Wi_new, torch.log(sigma_2_new) * torch.ones_like(self.log_Psi[i])

        new_params = [torch.stack(t) for t in zip(*[per_component_m_step(i) for i in range(k)])]

        def clip_pi(pi, min_weight=1e-3):
            # Ensure no weight is below the minimum
            clipped_pi = torch.clamp(pi, min=min_weight)
            # Renormalize the weights to sum to 1
            normalized_pi = clipped_pi / clipped_pi.sum(dim=-1, keepdim=True)
            return normalized_pi

        mu = new_params[0]
        W = new_params[1]
        log_Psi = new_params[2]
        pi = r_sum / torch.sum(r_sum) / torch.sum(weights)
        normalized_pi = clip_pi(pi)
        pi_logits = torch.log(normalized_pi)

        return [mu, W, log_Psi, pi_logits]
    
    def initialization(self, x:torch.Tensor, k:int) -> torch.Tensor:
        """
        Randomly initialize cluster assignments.

        Args:
            x (torch.Tensor): (n, d) tensor of input data.
            k (int): number of mixture components.

        Returns:
            r (torch.Tensor): tensor of per-component responsibilities.
        """

        # random initialization
        n = x.shape[0]
        idx = torch.randint(n, (k,))
        m = x[idx, :]
        
        # assign data to components
        similarity = x @ m.T - torch.sum(m * m, dim=1)[None, :] / 2
        label = torch.argmax(similarity, dim=1)
        # ensure unique labels
        u = torch.unique(label)
        while k != len(u):
            idx = torch.randint(n, (k,))
            m = x[idx, :]
            similarity = x @ m.T - torch.sum(m * m, dim=1)[None, :] / 2
            label = torch.argmax(similarity, dim=1)
            u = torch.unique(label)

        # create one-hot responsibility tensor
        r = torch.zeros(n, k, dtype=torch.int)
        r[torch.arange(n), label] = 1

        return r
    
    def fit(self, x:torch.Tensor, weights:torch.Tensor, k:int):
        """
        Fit the model parameters using expectation-maximization.

        Args:
            x (torch.Tensor): (n, d) tensor of input data.
            weights (torch.Tensor): (n,) tensor of importance weights.
            k (int): number of mixture components.
        """

        weights = weights[:,None]

        r = self.initialization(x, k)

        tol       = 1e-5
        maxiter   = 200
        llh       = torch.full([maxiter],-torch.inf)
        converged = False
        t         = 0

        # soft EM algorithm
        while (not converged) and (t+1 < maxiter):
            t = t+1
            # M-step
            [mu, W, log_Psi, pi_logits] = self.maximization(x, weights, r)
            self.mu = mu
            self.W = W
            self.log_Psi = log_Psi
            self.pi_logits = pi_logits
            # E-step
            [r, llh[t]] = self.expectation(x, weights)

            if t > 1:
                diff = llh[t]-llh[t-1]
                eps = abs(diff)
                converged = (eps < tol*abs(llh[t]) )

        if converged:
            print('Converged in', t,'steps.')
        else:
            print('Not converged in ', maxiter, ' steps.')

        return self
