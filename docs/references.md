# References

Academic citations for the algorithms, operators, and design decisions in this toolkit.

---

## Origin and Inspiration

### Ludobots

This toolkit was created in the context of work inspired by Josh Bongard's Ludobots evolutionary robotics course at the University of Vermont — the first MOOC run entirely on Reddit (2014). The course uses Pyrosim to co-evolve morphology and neural control of simulated robots, and provided the conceptual foundation for the Evolutionary-Robotics project from which this toolkit was extracted.

> Bongard, J. C. (2014). **Ludobots: An Online Evolutionary Robotics Course.** Reddit r/ludobots, University of Vermont.
> https://www.reddit.com/r/ludobots/

The intellectual basis for the course:

> Bongard, J. C. (2013). **Evolutionary Robotics.** *Communications of the ACM*, 56(8), 74–85.
> DOI: [10.1145/2493883](https://doi.org/10.1145/2493883)

### Foundational Texts

> Nolfi, S. and Floreano, D. (2000). ***Evolutionary Robotics: The Biology, Intelligence, and Technology of Self-Organizing Machines.*** Cambridge, MA: MIT Press.
> ISBN: 978-0-262-14070-6

> Pfeifer, R. and Bongard, J. C. (2006). ***How the Body Shapes the Way We Think: A New View of Intelligence.*** Cambridge, MA: MIT Press.
> ISBN: 978-0-262-16239-5

### Morphology-Behavior Co-Evolution

> Bongard, J., Zykov, V., and Lipson, H. (2006). **Resilient Machines Through Continuous Self-Modeling.** *Science*, 314(5802), 1118–1121.
> DOI: [10.1126/science.1133687](https://doi.org/10.1126/science.1133687)

> Bongard, J. C. (2011). **Morphological Change in Machines Accelerates the Evolution of Robust Behavior.** *Proceedings of the National Academy of Sciences*, 108(4), 1234–1239.
> DOI: [10.1073/pnas.1015390108](https://doi.org/10.1073/pnas.1015390108)

> Lipson, H. and Pollack, J. B. (2000). **Automatic Design and Manufacture of Robotic Lifeforms.** *Nature*, 406(6799), 974–978.
> DOI: [10.1038/35023115](https://doi.org/10.1038/35023115)

---

## Algorithms

### Differential Evolution (DE)

> Storn, R. and Price, K. (1997). **Differential Evolution — A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces.** *Journal of Global Optimization*, 11(4), 341–359.
> DOI: [10.1023/A:1008202821328](https://doi.org/10.1023/A:1008202821328)

Our implementation uses the DE/rand/1/bin variant described in this paper: random base vector, one difference vector, binomial crossover.

### CMA-ES

> Hansen, N. and Ostermeier, A. (2001). **Completely Derandomized Self-Adaptation in Evolution Strategies.** *Evolutionary Computation*, 9(2), 159–195.
> DOI: [10.1162/106365601750190398](https://doi.org/10.1162/106365601750190398)

> Hansen, N. (2016). **The CMA Evolution Strategy: A Tutorial.** *arXiv preprint* arXiv:1604.00772.
> DOI: [10.48550/arXiv.1604.00772](https://doi.org/10.48550/arXiv.1604.00772)

Our implementation follows the tutorial's reference algorithm: (mu/mu_w, lambda)-CMA-ES with weighted recombination, cumulative step-size adaptation (CSA), and rank-one + rank-mu covariance updates.

### Novelty Search

> Lehman, J. and Stanley, K. O. (2011). **Abandoning Objectives: Evolution Through the Search for Novelty Alone.** *Evolutionary Computation*, 19(2), 189–223.
> DOI: [10.1162/EVCO_a_00025](https://doi.org/10.1162/EVCO_a_00025)

The `NoveltySeeker` algorithm uses k-nearest-neighbor novelty scoring in behavioral space, following this paper's core insight that searching for novel behaviors rather than fitness can discover solutions that objective-driven search misses entirely.

---

## Operators

### Simulated Binary Crossover (SBX)

> Deb, K. and Agrawal, R. B. (1995). **Simulated Binary Crossover for Continuous Search Space.** *Complex Systems*, 9(2), 115–148.
> https://www.complex-systems.com/abstracts/v09_i02_a02/

SBX simulates single-point crossover from binary GAs in continuous space using a polynomial probability distribution controlled by the distribution index eta.

### NSGA-II (context for SBX and tournament selection)

> Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). **A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II.** *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.
> DOI: [10.1109/4235.996017](https://doi.org/10.1109/4235.996017)

SBX and tournament selection are the standard operators in NSGA-II. Our `TournamentSelection` and `SBXCrossover` follow the parameter conventions established here.

---

## Benchmarks

### ZDT Test Problems

> Zitzler, E., Deb, K., and Thiele, L. (2000). **Comparison of Multiobjective Evolutionary Algorithms: Empirical Results.** *Evolutionary Computation*, 8(2), 173–195.
> DOI: [10.1162/106365600568202](https://doi.org/10.1162/106365600568202)

The six ZDT functions (named after the authors' initials) are the standard benchmark suite for multiobjective optimization. Our `ZDT1Fitness` implements the first problem: convex Pareto front with 30 decision variables.

---

## Analytical Framework

### Beer's Dynamical Systems Approach

> Beer, R. D. (1995). **A Dynamical Systems Perspective on Agent-Environment Interaction.** *Artificial Intelligence*, 72(1–2), 173–215.
> DOI: [10.1016/0004-3702(94)00005-L](https://doi.org/10.1016/0004-3702(94)00005-L)

> Beer, R. D. (1995). **On the Dynamics of Small Continuous-Time Recurrent Neural Networks.** *Adaptive Behavior*, 3(4), 469–509.
> DOI: [10.1177/105971239500300405](https://doi.org/10.1177/105971239500300405)

The Evolutionary-Robotics project uses Beer's framework to analyze robot behavior as dynamical system trajectories. The four-pillar analytics (displacement, phase structure, spectral content, energetics) derive from this perspective. The `CliffMapper` and `LandscapeAnalyzer` tools detect the same behavioral cliffs that Beer's analysis reveals.

### POSIWID Principle

> Beer, S. (1974). ***Designing Freedom.*** CBC Massey Lectures. Toronto: House of Anansi Press.

"The purpose of a system is what it does" — Stafford Beer's operational definition of system purpose. Used in the Zimmerman toolkit's `POSIWIDAuditor` to measure gaps between intended and actual simulator outcomes.

---

## Simulation Infrastructure

### PyBullet

> Coumans, E. and Bai, Y. (2016–2021). **PyBullet, a Python Module for Physics Simulation for Games, Robotics and Machine Learning.** http://pybullet.org

The Evolutionary-Robotics project simulates 3-link robots in PyBullet at 240 Hz for 4000 timesteps per trial. The ea-toolkit was extracted from the optimization layer that sits above this simulation engine.

---

## Zimmerman Toolkit

### Thesis

> Zimmerman, J. E. (2025). ***Locality, Relation, and Meaning Construction in Language, as Implemented in Humans and Large Language Models (LLMs).*** PhD dissertation, University of Vermont.

The Zimmerman toolkit implements the dissertation's framework for LLM-mediated simulation interrogation: diegetic vs. supradiegetic content (§2.2.3, §3.5.3), Power-Danger-Structure dimensions (§4.6.4), TALOT/OTTITT meaning-from-contrast (§4.7.6), and POSIWID alignment auditing (§3.5.2). See [`zimmerman_bridge`](zimmerman_bridge.md) for how the two toolkits integrate.
