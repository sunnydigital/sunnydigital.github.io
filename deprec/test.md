---
title: TEST
date: 2023-07-23T07:00:00-04:00
draft: true
ShowToc: true
math: true
cover:
    relative: true
---
$$
\sum_{i=1}^{N} || \mathbf{x}_i - f_G (\mathbf{z}_i) ||^2
$$

$$
\frac{\lambda}{2} \sum_{(V_i, V_j) \in \mathcal{E}} b_{i,j}  || f_G (\mathbf{z}_i) - f_G (\mathbf{z}_j) ||^2
$$

$$
\mathcal{G}{}_{b_{\mathcal{g}}}
$$

???:
$$
\mathcal{z}_i
$$

$$
\sum_{(V_i, V_j) \in \mathcal{E}} b_{i,j} 
$$

$$
\min_{\mathcal{G} \in G_b}{} \min_{f_{\mathcal{G}} \in \mathcal{F}}{} \min_{Z}{} \sum_{(V_i, V_j) \in \mathcal{E}} b_{i,j}  || f_G (\mathbf{z}_i) - f_G (\mathbf{z}_j) ||^2
$$

$$
\min_{\mathcal{G} \in \mathcal{G}_{b}}{} \min_{f_{\mathcal{G}} \in \mathcal{F}}{} \min_{Z}{}
$$

???:
$$
\min_{\mathcal{G} \in G_b}{} \min_{f_{\mathcal{G}} \in \mathcal{F}}{} \min_{Z}{}
$$

???:
$$
\min_{\mathcal{G} \in \mathcal{G}_{b}}{} \min_{\mathcal{f}_{\mathcal{G}} \in \mathcal{F}}{} \min_{Z}{} \sum_{(V_i, V_j) \in \mathcal{E}} b_{i,j} {|| f_{\mathcal{G}}(\mathbf{z}_i) - f_{\mathcal{G}}(\mathbf{z}_{j}) ||}^2
$$

Works:
$$
\min_{\mathcal{X} \in \mathcal{X}}{} \min_{\mathcal{Y} \in \mathcal{Y}}{} \min_{\mathcal{Z} \in \mathcal{Z}}{}
$$

Works:

$$
\min_{\mathcal{G} \in \mathcal{G}_{b}}
$$

Works:

$$
\min_{\mathcal{f}_{\mathcal{G}} \in \mathcal{F}}
$$

Works:

$$
\min_{Z}
$$

Doesn't Work:

$$
\min_{\mathcal{G} \in \mathcal{G}_{b}}
\min_{\mathcal{f}_{\mathcal{G}} \in \mathcal{F}}
$$

Doesn't Work:

$$
\min_{\mathcal{G} \in \mathcal{G}_{b}},
\min_{\mathcal{f}_{\mathcal{G}} \in \mathcal{F}},
\min_{Z}
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
\min_{\mathcal{G} \in \mathcal{G}_{b}}, \min_{\mathcal{f}_{\mathcal{G}} \in \mathcal{F}}, \min_{Z}, \sum_{(V_i, V_j) \in \mathcal{E}} b_{i,j} \mathcal{f}_{\mathcal{G}}(\mathbf{z}_i)-\mathcal{f}_{\mathcal{G}}(\mathbf{z}_{j})
$$

Doesn't Work:

$$
\begin{equation}
\operatorname{min}_{G \in G_{b}} \operatorname{min}_{f_{G} \in F} \operatorname{min}_{Z} \sum_{(V_i, V_j) \in E}
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