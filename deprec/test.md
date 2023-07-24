---
title: TEST
date: 2023-07-23T07:00:00-04:00
draft: false
ShowToc: true
math: true
cover:
    relative: true
---
Works:

$$
\argmin_{\mathcal{G} \in \mathcal{G}_{b}}
$$

Works:

$$
\argmin_{\mathcal{f}_{\mathcal{G}} \in \mathcal{F}}
$$

Works:

$$
\argmin_{Z}
$$

Doesn't Work:

$$
\argmin_{\mathcal{G} \in \mathcal{G}_{b}}
\argmin_{\mathcal{f}_{\mathcal{G}} \in \mathcal{F}}
$$

Doesn't Work:

$$
\argmin_{\mathcal{G} \in \mathcal{G}_{b}},
\argmin_{\mathcal{f}_{\mathcal{G}} \in \mathcal{F}},
\argmin_{Z}
$$

Works:

$$
 \sum_{(V_i, V_j) \in \mathcal{E}}
$$

Doesn't Work:

$$
b_{i,j} ||\mathcal{f}_{\mathcal{G}}(\mathbf{z}_i)-\mathcal{f}_{\mathcal{G}}(\mathbf{z}_{j})||^2
$$

Doesn't Work:

$$
\argmin_{\mathcal{G} \in \mathcal{G}_{b}}, \argmin_{\mathcal{f}_{\mathcal{G}} \in \mathcal{F}}, \argmin_{Z}, \sum_{(V_i, V_j) \in \mathcal{E}} b_{i,j} \mathcal{f}_{\mathcal{G}}(\mathbf{z}_i)-\mathcal{f}_{\mathcal{G}}(\mathbf{z}_{j})
$$

Doesn't Work:

$$
\begin{equation}
\operatorname{argmin}_{G \in G_{b}} \operatorname{argmin}_{f_{G} \in F} \operatorname{argmin}_{Z} \sum_{(V_i, V_j) \in E}
\end{equation}
$$

Works:

$$
F_{X}(x)=\operatorname{Pr}(X \leq x)= \begin{cases} 
0 &\text{if } x<0 \\\\ 
\frac{1}{2} &\text{if } 0 \leq x<1 \\\\ 
1 &\text{if } x \geq 1 \end{cases}
$$

Doesn't Work:

$$
F_{X}(x)=\operatorname{Pr}(X \leq x)= 
\left\{ \begin{array}{l}
0 \text { if } x<0 \\
\frac{1}{2} \text { if } 0 \leq x<1 \\
1 \text { if } x \geq 1
\end{array} \right.
$$

Works:

$$
\begin{aligned}
P\left(X=x | Y=c_{k}\right) &=P\left(X^{(1)}=x^{(1)}, \cdots, X^{(n)}=x^{(n)} | Y=c_{k}\right) \\\\ 
&=\prod_{j=1}^{n} P\left(X^{(j)}=x^{(j)} | Y=c_{k}\right)
\end{aligned}
$$

Works:

$$
\begin{aligned} P\left(X=x | Y=c_{k}\right) &=P\left(X^{(1)}=x^{(1)}, \cdots, X^{(n)}=x^{(n)} | Y=c_{k}\right) \\\\ 
&=\prod_{j=1}^{n} P\left(X^{(j)}=x^{(j)} | Y=c_{k}\right) \end{aligned}
$$