# PPO with adaptive exploration

## Algorithm
The methods for adaptive exploration we developed are used in the context of PPO algorithms. A traditional objective of a PPO algorithm is the one introduced by 
OpenAI, represented by the following loss function:

\begin{equation}
    L_t(\theta) = \hat{\mathbb{E}}_t\left[L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t)\right]
\end{equation}

Our focus is on the entropy bonus represented by $S$ and the entropy coefficient $c_2$, which determines the exploration magnitude. In traditional PPO implementations, $c_2$ remains a fixed hyperparameter throughout training. This paper introduces a new learning algorithm, PPO with adaptive exploration (axPPO), which incorporates a dynamic scaling of the entropy coefficient based on the recent return ($G_{recent}$) obtained by the agent:

\begin{equation}
    L_t(\theta) = \hat{\mathbb{E}}_t\left[L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + G_{recent} \,c2 \,S[\pi_\theta](s_t)\right]
\end{equation}

The adaptive exploration framework relies on $G_{recent}$, computed as:

\begin{equation}
    G_{recent} = G_t(\tau) = \frac{1}{G_{max}}\frac{\sum_{i=t-\tau}^{t} \overline{G}_i^{batch}}{\tau}
\end{equation}

Here, $t$ represents the current time step, $\tau$ is a time constant determining how far back to integrate past returns, and $\overline{G}_i^{batch}$ is the mean return across batches at time step $i$. The result is scaled between 0 and 1 by dividing it by $G_{max}$, the maximum return an agent can receive during an episode. In short, $G_{recent}$ represents the recent performance of the agent, parameterized by $\tau$.

```
