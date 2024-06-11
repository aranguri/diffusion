**Summary Table**

for $x_1\sim \eta + s\mu$ with $\eta\sim \text{N}(0,\sigma^2 I), s \sim \text{Ber}(p)$

inter($p$) / intra ($\sigma$) / intra ($\mu$)

| variance | order of $b$ | Math | Experiment| Comment         |
|----------|------------|------|------|-----------------|
| P       | 0          | no/yes/yes | no/yes/yes  |   from paper    |
| P   | 1          | no/yes/yes | no/yes/yes      |  $b$ vanishes   |
| P   | $d$        | ?/?/?      | ?/?/?       |                 |
| E   | 1          | ?/no/?     | ?/?/?       |         |


Paper
https://arxiv.org/pdf/2310.03575

Stat Phys
https://sphinxteam.github.io/EPFLDoctoralLecture2022/Notes.pdf

Stoch Interpolants
https://arxiv.org/pdf/2303.08797

Useful results

$$
\mathbb E[x_1|x_t] = \frac{\beta_t}{\alpha_t^2 + \beta_t^2} x_t + \frac{\alpha_t^2}{\alpha_t^2 + \beta_t^2}\mu \tanh\left(\frac{\beta_t}{\alpha_t^2 + \beta_t^2} \langle x_t,\mu\rangle + \frac{1}{2}\log\left(\frac{p}{1-p}\right)\right)
$$

Plan
p =.9
arch (both dividing by d and sqrt(d))
*std = np.sqrt(d) all with bias
 * u and w as weights (not coupled)
 * coupled, with scalar d_t
 * coupled no scalar
*no bias