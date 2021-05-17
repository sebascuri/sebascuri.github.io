---
layout: post
title: Combining Pessimism with Optimism for Robust and Efficient Model-Based Deep Reinforcement Learning
categories: [Reinforcement Learning]
---

## Introduction

In this work, we address the challenge of finding a robust policy in continuous control tasks.
As a motivating example, consider designing a braking system on an autonomous car. 
As this is a highly complex task, we want to learn a policy that performs this maneuver.
One can imagine many real-world conditions and simulate them during training time e.g., road conditions, brightness, tire pressure, laden weight, or actuator wear. 
However, it is unfeasible to simulate all such conditions and these might vary in potentially unpredictable ways. 
The main goal is to learn a policy that provably brakes in a robust fashion so that, even when faced with new conditions, it performs reliably.

Of course, the first key challenge is to model the possible conditions the system might be in. 
Inspired in \\(\mathcal{H}_\infty\\) control and Markov Games, we model unknown conditions with an adversary that interacts with the environment and the agent. 
The adversary is allowed to execute adversarial actions at every instant by observing the current state and the history of the interaction. For example, if one wants to be robust to laden weight changes in the braking example, the adversary has the ability to choose the laden weight. 
By ensuring good performance with respect to the **worst-case** laden weight that an adversary selects, then we may also ensure good performance with respect to **any other** laden weight, and we say that **such policy is robust**. 


The main algorithmic procedure is to train our agent together with a ficticious adversary that simulates the real-world conditions we want to be robust to.
The key question we ask is **how to train these agents in a data-efficient way**. 
A common procedure is domain randomization, in which the adversary is simulated by sampling at random from a domain distribution. This has proved useful for sim2real applications, but it has the drawback that it requires many samples from the domain, as easy samples are treated equally than hard samples, and it scales poorly with the dimensionality of the domain.
Another approach is RARL, in which the adversary and the learner are trained through greedy gradient descent, i.e., without any exploration. 
Although it performs well in some tasks, we demonstrate that the lack of exploration, particularly by the adversary, yields poor performance. 


## Problem Setting

In this section, we formalize the ideas from the introduction. 

We consider a stochastic environment with states $s$, agent actions $a$, adversary actions $\bar{a}$, and i.i.d. additive transition noise vector $\omega$. 
The dynamics are given by:
\begin{equation}
    s_{h+1} = f(s_h, a_h, \bar{a}_h) + \omega_h
\end{equation}
We assume the true dynamics $f$ are **unknown** and consider the episodic setting over a finite time horizon $H$.
After every episode (i.e., every $H$ time steps), the system is reset to a known state $s_0$.
At every time-step, the system returns a reward $r(s_h, a_h, \bar{a}_h)$
We consider time-homogeneous policies agent policies $\pi$, that select actions according to $a_h = \pi(s_h)$, as well as
adversary policies $\bar{\pi}$ with $\bar{a}_h = \bar{\pi}(s_h)$.



We consider the performance of a pair of policies $(\pi, \bar{\pi})$ on a given dynamical system $\tilde{f}$ as the episodic expected sum of returns:
 $$
 \begin{align}
    J(\tilde{f}, \pi, \bar{\pi}) &:= \mathbb{E}_{\tau_{\tilde{f}, \pi, \bar{\pi}}}{ \left[ \sum_{h=0}^{H} r(s_h, a_h, \bar{a}_h) \, \bigg| \, s_0 \right]}, \\
    \text{s.t. }\;  s_{h+1}& = \tilde{f}(s_h,  a_h, \bar{a}_h) + \omega_h, \nonumber
\end{align}
$$

where $\tau_{\tilde{f},\pi,\bar{\pi}}$ is a random trajectory induced by the stochastic noise $\omega$, the dynamics $\tilde{f}$, and the policies $\pi$ and $\bar{\pi}$. 


We use $\pi^{\star}$ to denote the optimal robust policy from the set $\Pi$ on the true dynamics $f$, i.e.,
\begin{equation}
    \pi^\star \in \arg \max_{\pi \in \Pi} \min_{\bar{\pi} \in \bar\Pi} J(f, \pi, \bar\pi). \label{eq:objective}
\end{equation}

For a small fixed $\epsilon>0$, the goal is to output a robust policy $\pi_{T}$ after $T$ episodes such that:
\begin{equation}
    \min_{\bar{\pi} \in \bar{\Pi}} J(f, \pi_T, \bar{\pi}) \geq \min_{\bar{\pi} \in \bar{\Pi}} J(f, \pi^{\star}, \bar{\pi}) - \epsilon,
\end{equation}
Hence, we consider the task of near-optimal robust policy identification. Thus, the goal is to output the agent's policy with near-optimal robust performance when facing its own **worst-case** adversary, and the adversary selects $\bar{\pi}$ **after** the agent selects $\pi_T$. 
This is a stronger robustness notion than just considering the worst-case adversary of the optimal policy, since, by letting $\bar{\pi}^* \in  \arg \min_{\bar{\pi} \in \bar\Pi} J(f, \pi^{\star}, \bar\pi)$, we have $J(f, \pi_T, \bar\pi^*) \geq \min_{\bar\pi \in \bar\Pi} J(f, \pi_T, \bar\pi)$.

## Method: RH-UCRL

In this section, we present our algorithm, RH-UCRL. It is a model-based algorithm that explicitly uses the **epistemic uncertainty** in the model to explore, both for the agent and for the adversary. 
In the next subsection, I will explain how we do the model-learning procedure, and in the following one, the policy optimization. 

### Model Learning

RH-UCRL is agnostic to the model one decides to use, as long as it is able to distinguish between epistemic and aleatoric uncertainty. In this work, we decided to use probabilistic ensembles of neural networks as in PETS and H-UCRL. 


### Policy Optimization


## Applications
### Domain Randomization

### Action Robust Reinforcement Learning

### Robust Reinforcement Learning