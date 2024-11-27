#import "setup.typ": *
#show: notes-template

= Domain

In the most general sense, $Omega$ may be a (piecewise) smooth oriented and
bounded $n$-dimensional Riemannian manifold, $n in NN$, with a piecewise smooth
boundary.

= Simplicial Manifold

The dimension is $n$.

A *simplicial manifold* is a *simplicial complex* for which the geometric
realization is homeomorphic to a topological manifold.

For a simplicial complex to manifold, the neighborhood of each vertex (i.e. the
set of simplices that contain that point as a vertex) needs to be homeomorphic
to a $n$-ball.

- The $n$-simplicies of the manifold are called cells.
- The $n-1$-simplicies of the manifold are called facets.

= Geometry of Simplicial Manifold

Geometric information is conveyed through a Riemannian metric, turning our
object of study into a *simplicial Riemannian manifold*.

It's the central object in the study of *Regge Calculus*.
Here the Riemannian metric is called the *Regge metric* and is fully specified
by knowing the edge lenghs of the simplicial complex.

= Simplicial Homology

Simplicial complex is extended to simplicial chain complex.

= Differential Form

- $k$-dimensional ruler $omega in Lambda^k (Omega)$
- ruler $omega: p in Omega |-> omega_p$ varies continuously  across manifold according to coefficent functions.
- locally measures tangential $k$-vectors $omega_p: (T_p M)^k -> RR$
- globally measures $k$-dimensional submanifold $integral_M omega in RR$

$
  phi: [0,1]^k -> Omega
  quad quad
  M = "Image" phi
  \
  integral_M omega =
  limits(integral dots.c integral)_([0,1]^k) quad
  omega_(avec(phi)(t))
  ((diff avec(phi))/(diff t_1) wedge dots.c wedge (diff avec(phi))/(diff t_k))
  dif t_1 dots dif t_k
$

#pagebreak()
= Vector Calculus

== Derivatives
$
  grad f &= (dif f)^sharp
  \
  "curl" avec(F) &= hodge dif hodge avec(F)^flat
  \
  "div" avec(F) &= hodge dif avec(F)^flat
$


== Main Theorems

// Gradient Theorem
$
  integral_C grad f dot dif avec(s) =
  phi(avec(b)) - phi(avec(a))
$
// Curl Theorem
$
  integral.double_S curl avec(F) dot dif avec(S) =
  integral.cont_(diff A) avec(F) dot dif avec(s)
$
// Divergence Theorem
$
  integral.triple_V "div" avec(F) dif V =
  integral.surf_(diff V) avec(F) dot nvec(n) dif A
$

== Sobolev Spaces

$
  H (grad; Omega) &= { u: Omega -> RR : &&grad u in [L^2 (Omega)]^3 }
  \
  Hvec (curl; Omega) &= { avec(u): Omega -> RR^3 : &&curl u in [L^2 (Omega)]^3 }
  \
  Hvec (div ; Omega) &= { avec(u): Omega -> RR^3 : &&div u in L^2 (Omega) }
$


#grid(
  columns: (40%, 60%),
  align: horizon,
  $
    &H    (grad; Omega) &&supset.eq cal(S)^0_1   (mesh) \
    &Hvec (curl; Omega) &&supset.eq bold(cal(N))   (mesh) \
    &Hvec (div ; Omega) &&supset.eq bold(cal(R T)) (mesh) \
  $,
  [
    - Lagrangian basis on vertices $Delta_0 (mesh)$
    - Nédélec basis on edges $Delta_1 (mesh)$
    - Raviart-Thomas basis on faces $Delta_2 (mesh)$
  ]
)

$
  &H    (grad; Omega) &&=^~ H Lambda^0 (Omega) \
  &Hvec (curl; Omega) &&=^~ H Lambda^1 (Omega) \
  &Hvec (div ; Omega) &&=^~ H Lambda^2 (Omega) \
$


Vector Calculus de Rham Complex in 3D
$
  0 -> H (grad; Omega) limits(->)^grad Hvec (curl; Omega) limits(->)^curl Hvec (div; Omega) limits(->)^div L^2(Omega) -> 0
  \
  curl compose grad = 0
  quad quad
  div compose curl = 0
$


$
  0 -> cal(S)^0_1 (mesh) limits(->)^grad bold(cal(N)) (mesh) limits(->)^curl bold(cal(R T)) (mesh) limits(->)^div cal(S)^(-1)_0 (mesh) -> 0
$



= Main Definition and Theorems of Exterior Calculus

Stokes' Theorem
$
  integral_Omega dif omega = integral_(diff Omega) omega
$
for all $omega in Lambda^l_1 (Omega)$

A fundamental fact about exterior differentiation is that $dif(dif omega) = 0$
for any sufficiently smooth differential form $omega$.

Under some restrictions on the topology of $Omega$ the converse is
also true, which is called the exact sequence property:

*Theorem (Poincaré's lemma):*
For a contractible domain $Omega subset.eq RR^n$ every
$omega in Lambda^l_1 (Omega), l >= 1$, with $dif omega = 0$ is the exterior
derivative of an ($l − 1$)–form over $Omega$.

A second main result about the exterior derivative is the following formula

*Theorem (Integration by parts):*
$
  integral_Omega dif omega wedge eta
  + (-1)^l integral_Omega omega wedge dif eta
  = integral_(diff Omega) omega wedge eta
$
for $omega in Lambda^l (Omega), eta in Lambda^k (Omega), 0 <= l, k < n − 1, l + k = n − 1$.

Here, the boundary $diff Omega$ is endowed with the induced orientation.
Finally, we recall the pullback $Omega |-> Phi^*omega$ under a change of
variables described by a diffeomorphism $Phi$. This transformation commutes
with both the exterior product and the exterior derivative, and it leaves the
integral invariant.


Remember that given a basis
$dif x_0, dots, dif x_n$ of the dual space of $T_Omega (xv)$ the set of all
exterior products of these
furnishes a basis for the space of alternating l-multilinear forms on
TΩ(x). Thus any ω ∈ Dl(Ω) has a representation
$
  omega = sum_(i_1, dots, i_l) phi_(i_1,dots,i_l) dif x_i_1 wedge dots.c wedge dif x_i_l
$
where the indices run through all combinations admissible according
to (6) and the ϕi1,... ,il : Ω �→ R are coefficient functions.
Therefore, we call a differential form polynomial of degree k,
k ∈ N0, if all its coefficient functions in (7) are polynomials of degree k.


We can define proxies to convert between vector fields and differential forms.
Sharp #sharp to move from differential form to vector field.
Flat #flat to move from vector field to differential form.


== Stockes' Theorem
$
  integral_M dif omega = integral_(diff M) trace omega
$

Product rule / Leibniz rule
$
  dif (alpha wedge beta) = dif alpha wedge beta + (-1)^abs(alpha) alpha wedge dif beta
$

Definition Hodge star operator via inner product
$
  beta wedge (hodge alpha) = inner(alpha, beta) vol
$
where $vol = dif x^1 wedge dots wedge dif x^n$

To solve for $hodge alpha$ we need to solve the linear systems of equations, we obtain
from all different $beta$, the basis of $beta$ suffices to get a LSE.
If we just ignore the inner product and replace it by one, by can find $hodge alpha$ directly.

Definition Codifferential on $k$-forms in $n$-space.
$
  delta = (-1)^(n(k+1)+1) s hodge dif hodge = (-1)^k hodge^(-1) dif hodge
$

$
  "div" F = hodge dif hodge omega_F
  quad quad
  "curl" F = hodge dif omega_F
$


= Explicit Formula for Lagrangian FE in arbitrary dimensions

We are on a simplicial manifold of dimension $d$.
We recognize that for any $d$-simplex the vertices are always opposite of a unique
$(d-1)$-face.
- For a 1-simplex this is the other vertex.
- For a 2-simplex this is the opposite edge.
- For a 3-simplex this is the opposite triangle.

This opposite face has a unique normal vector (this is always a 1-vector since this is the orthogonal complement (hodge star) of a (d-1)-space in an outer d-space).
We have therefore a pairing between each vertex and the normal vector of it's opposite (d-1)-face.

The gradient of the barycentric coordinate function of a vertex is a (negatively) scaled normal vector of the opposite face.
What is the scale factor?

== Derivation of the formula

We derive a general formula for the element matrix of the laplacian for an arbitrary k-simplex $K$.
We express it in terms of geometric quantities such as volumes and angles.

- Let $F_i$​be the $(k-1)$-face opposite vertex $v_i$.
- Let $S_i$​be the $(k−1)$-measure of $F_i$.
- Let $n_i$​be the outward normal vector to $F_i$, scaled by $S_i$:
  $n_i = S_i hat(n)_i$
- The gradient $nabla lambda_i$ is then given by:
  $nabla lambda_i = (n_i)/(k abs(K))$

The element matrix entires are
$
  a_(i j) = integral_K nabla lambda_i dot nabla lambda_j dif x
$

Since the gradients are constant the integral simplifies to
$
  a_(i j) = 
  (nabla lambda_i dot nabla lambda_j) abs(K) =
  ((n_i)/(k abs(K)) dot (n_j)/(k abs(K))) abs(K) =
  (n_i dot n_j)/(k^2 abs(K))
$

The dot product $n_i dot n_j$ can be expressed using the dihedral angle $theta_(i j)$ between the faces $F_i$ and $F_j$.
$
  n_i dot n_j = S_i S_j cos theta_(i j)
$

Giving us the final formula
$
  a_(i j) = (S_i S_j cos theta_(i j))/(k^2 abs(K))
$


=== Whitney Forms

$
  cal(W) Lambda^k (mesh) = "span" {lambda_sigma : sigma in Delta_k (mesh)}
$

#grid(
  columns: (50%, 50%),
  align: center + horizon,
  [
    - $cal(W) Lambda^0 (mesh)$ on 0-simplices  $Delta_0 (mesh)$
    - $cal(W) Lambda^1 (mesh)$ on 1-simplicies $Delta_1 (mesh)$
    - $cal(W) Lambda^2 (mesh)$ on 2-simplicies $Delta_2 (mesh)$
  ],
  $
    cal(W) Lambda^0 (mesh) &=^~ cal(S)^0_1 (mesh) \
    cal(W) Lambda^1 (mesh) &=^~ bold(cal(N)) (mesh) \
    cal(W) Lambda^2 (mesh) &=^~ bold(cal(R T)) (mesh) \
  $,
)


The whitney $p$-form corresponding to the $p$-simplex $[x_0, dots, x_p]$ is
$
  cal(W) [x_0, dots, x_p] =
  p! sum_(i=0)^p (-1)^i lambda_i
  (dif lambda_0 wedge dots.c wedge hat(dif lambda_i) wedge dots.c wedge dif lambda_p)
$

As example in a triangle $K = [x_0,x_1,x_2]$ the whitney forms are
$
  cal(W) [x_0] = lambda_0
  wide
  cal(W) [x_1] = lambda_1
  wide
  cal(W) [x_2] = lambda_2
  \
  cal(W) [x_0,x_1] = lambda_0 dif lambda_1 - lambda_1 dif lambda_0
  wide
  cal(W) [x_0,x_2] = lambda_0 dif lambda_2 - lambda_2 dif lambda_0
  wide
  cal(W) [x_1,x_2] = lambda_1 dif lambda_2 - lambda_2 dif lambda_1
  \
  cal(W) [x_0,x_1,x_2] = 2 (lambda_0 (dif lambda_1 wedge dif lambda_2) - lambda_1 (dif lambda_0 wedge dif lambda_2) + lambda_2 (dif lambda_0 wedge dif lambda_1))
$

Whitney forms are affine invariant. \
Let $K_1 = [x_0, dots x_n]$ and $K_2 = [y_0, dots y_n]$ and $phi: K_1 -> K_2$ affine map, such that $phi(x_i) = y_i$,
then
$W[x_0, dots, x_n] = phi^* (W[y_0, dots y_n])$

== Various

Exterior derivative: $dif_l: dom(d_l) subset.eq L^2 Lambda^l (Omega) -> L^2 Lambda^(l+1)$

$L^2$-adjoint: $delta_l := dif _(l-1)^* = (-1)^l hodge_(l-1)^(-1) compose dif_(N-l) compose hodge_l$


= Hodge star Operator

The Hodge star operator is a linear operator
$
  hodge: Lambda^k (Omega) -> Lambda^(n-k) (Omega)
$
s.t.
$
  alpha wedge (hodge beta) = inner(alpha, beta) vol
$

In general:\
- $hodge 1 = vol$
- $hodge vol = 1$

with $vol = sqrt(abs(g)) dif x^1 dots dif x^n$



== DEC considerations

DEC	and	FEEC can benefit from	a closer interaction.
DEC	can	be obtained	from the lowest	order	FEEC by	replacing
the mass matrices	by ones	that are computed	from primal-dual meshes.

One distinguishing aspect of DEC is this geometric nature of the mass matrix.
The dual meshes are circumcentric duals and in the early years it was known
that simplices which contain their own circumcenters are sufficient for
producing appropriate mass matrices. In recent work we proved that the much
broader class of Delaunay meshes (modulo some boundary restrictions) suffice.

= Sobolev Spaces

$
  H Lambda^k (Omega) = { omega in L^2 Lambda^k (Omega) : dif omega in L^2 Lambda^(k+1) (Omega) }
$


= (Co-) Chain-Complexes
#v(1cm)

Continuous Chain Complex
$  
  0 limits(<-) C_0 (Omega) limits(<-)^diff dots.c limits(<-)^diff C_n (Omega) limits(<-) 0
  \
  diff^2 = diff compose diff = 0
$

Simplicial Chain Complex
$
  0 limits(<-) Delta_0 (mesh) limits(<-)^diff dots.c limits(<-)^diff Delta_n (mesh) limits(<-) 0
  \
  diff^2 = diff compose diff = 0
$

Vector Calculus de Rham Complex in 3D
$
  0 -> H (grad; Omega) limits(->)^grad Hvec (curl; Omega) limits(->)^curl Hvec (div; Omega) limits(->)^div L^2(Omega) -> 0
  \
  curl compose grad = 0
  quad quad
  div compose curl = 0
$


De Rham Complex
$
  0 -> H Lambda^0 (Omega) limits(->)^dif dots.c limits(->)^dif H Lambda^n (Omega) -> 0
  \
  dif^2 = dif compose dif = 0
$

Whitney Subcomplex
$
  0 -> cal(W) Lambda^0 (mesh) limits(->)^dif dots.c limits(->)^dif cal(W) Lambda^n (mesh) -> 0
$

Polynomial Subcomplex
$
  0 -> cal(P)_r Lambda^0 (mesh) limits(->)^dif dots.c limits(->)^dif cal(P)_r Lambda^n (mesh) -> 0
$


  //#diagram(
  //  edge-stroke: fgcolor,
  //  cell-size: 15mm,
  //  $
  //    0 edge(->) &H(grad; Omega) edge(grad, ->) &Hvec (curl; Omega) edge(curl, ->) &Hvec (div; Omega) edge(div, ->) &L^2(Omega) edge(->) &0
  //  $
  //)

= Maxwell


Relativistic Electrodynamics
- Maxwell's Equations on 4D Spacetime Manifold!
- Faraday 2-form $F = E wedge dif t + B$
- Current 3-form $J = rho + J wedge dif t$
$
  dif F = 0 \
  dif (star F) = J
$


= $L^2$-inner product

The inner product is defined as
$
  inner(omega, eta)_(L^2 Lambda^k)
  =
  integral_Omega inner(omega_x, eta_x) "vol"
  =
  integral omega wedge star eta
$


= Coderivative Operator

Coderivative operator $delta: Lambda^k (Omega) -> Lambda^(k-1) (Omega)$
defined such that
$
  star delta omega = (-1)^k dif star omega
$

= Hodge-Laplace Problem

$
  Delta u = f
$
with $u,f in Lambda^k (Omega)$

Hodge-Laplace operator
$
  Delta: Lambda^k (Omega) -> Lambda^k (Omega)
  \
  Delta = dif delta + delta dif
$

= Weak Variational Form
Form the $L^2$-inner product with a test "function" $v in Lambda^k (Omega)$.
$
  Delta u = f
$

We obtain the variational equation
$
  u in H Lambda^k (Omega): quad quad
  inner(Delta u, v) = inner(f, v)
  quad quad forall v in H Lambda^k (Omega)
$

Or in integral form
$
  integral_Omega ((dif delta + delta dif) u) wedge star v = integral_Omega f wedge star v
$

= Integration by Parts

$
  integral_Omega dif omega wedge eta
  =
  (-1)^(k-1)
  integral_Omega omega wedge dif eta
  +
  integral_(diff Omega) "Tr" omega wedge "Tr" eta
$

$
  inner(dif omega, eta) = inner(omega, delta eta) + integral_(diff Omega) "Tr" omega wedge "Tr" star eta
$

If $omega$ or $eta$ vanishes on the boundary, then
$delta$ is the formal adjoint of $dif$ w.r.t. the $L^2$-inner product.
$
  inner(dif omega, eta) = inner(omega, delta eta)
$

$
  inner(Delta u, v) = inner(f, v)
  \
  inner((dif delta + delta dif) u, v) = inner(f, v)
  \
  inner((dif delta + delta dif) u, v) = inner(f, v)
  \
  inner(dif delta u, v) + inner(delta dif u, v) = inner(f, v)
  \
  inner(delta u, delta v) + inner(dif u, dif v) = inner(f, v)
$

#v(1cm)

$
  u in H Lambda^k (Omega): quad quad
  inner(delta u, delta v) + inner(dif u, dif v) = inner(f, v)
  quad
  forall v in H Lambda^k (Omega)
$

$
  u in H Lambda^k (Omega): quad
  integral_Omega (delta u) wedge star (delta v) + integral_Omega (dif u) wedge star (dif v) = integral_Omega f wedge star v
  quad
  forall v in H Lambda^k (Omega)
$

= Galerkin Discretization
$
  u_h = sum_(i=1)^N mu_i phi_i
  quad quad
  v_h = phi_j
$

$
  u in H Lambda^k (Omega): quad quad
  inner(delta u, delta v) + inner(dif u, dif v) = inner(f, v)
  quad quad forall v in H Lambda^k (Omega)
$
$
  vvec(mu) in RR^N: quad
  sum_(i=1)^N mu_i (integral_Omega (delta phi_i) wedge star (delta phi_j) + integral_Omega (dif phi_i) wedge star (dif phi_j))
  =
  sum_(i=1)^N mu_i integral_Omega f wedge star phi_j
  quad forall j in {1,dots,N}
$

$
  amat(A) vvec(mu) = 0
  \
  A =
  [integral_Omega (delta phi_i) wedge star (delta phi_j)]_(i,j=1)^N
  +
  [integral_Omega (dif phi_i) wedge star (dif phi_j)]_(i,j=1)^N
  \
  vvec(phi) = [integral_Omega f wedge star phi_j]_(j=1)^N
$

= Exterior Derivative
Purely topological, no geometry.

In discrete settings defined as coboundary operator, through Stokes' theorem.\
So the discrete exterior derivative is just the transpose of the boundary operator / incidence matrix.

= Hodge Star operator
#v(1cm)

Defined such that:
$ alpha wedge star alpha = "vol" $

= Whitney Forms

The interpolation of a discrete cochain is a piecewise-linear differential form. \
A so-called Whitney form. \
The interpolation basis is the Whitney Basis.

Whitney Forms are constructed from barycentric coordinate functions $lambda_i$.

Hat means omitted.
$
  cal(W)[v_i] = lambda_i
  \
  cal(W)[v_0,dots,v_k] = k! sum_(i=0)^k (-1)^i lambda_i (dif lambda_0 wedge dots.c wedge hat(dif lambda_i) wedge dots.c lambda_k)
$

Some expansions:
$
  cal(W)[v_0 v_1] =
  &lambda_0 dif lambda_1 - lambda_1 dif lambda_0
  \
  cal(W)[v_0 v_1 v_2] =
    &2 lambda_0 (dif lambda_1 wedge dif lambda_2) \
  - &2 lambda_1 (dif lambda_0 wedge lambda_2) \
  + &2 lambda_2 (dif lambda_0 wedge dif lambda_1) \
  \
  cal(W)[v_0,v_1,v_2,v_3] =
    &6 lambda_0 (dif lambda_1 wedge dif lambda_2 wedge dif lambda_3) \
  - &6 lambda_1 (dif lambda_0 wedge lambda_2 wedge lambda_3) \
  + &6 lambda_2 (dif lambda_0 wedge dif lambda_1 wedge dif lambda_3) \
  - &6 lambda_3 (dif lambda_0 wedge dif lambda_1 wedge dif lambda_2) \
$

The defining property of the Whitney basis is:\
For any two simplicies $S, T in Delta_k (mesh)$, we have
$
  integral_S cal(W)_T = cases(
    +1 quad &"if" S = T,
    -1 quad &"if" S = -T,
    0  quad &"if" S != T,
  )
$

This is like a generalized Lagrange basis property.

= Barycentric Coordinates

Barycentric coordinates exist for all $k$-simplicies.\
They are a coordinate system relative to the simplex.\
Where the barycenter of the simplex has coordinates $(1/k)^k$.

$
  x = sum_(i=0)^k lambda^i (x) space v_i
$
with $sum_(i=0)^k lambda_i(x) = 1$ and inside the simplex $lambda_i (x) in [0,1]$ 

$
  lambda^i (x) = det[v_0,dots,v_(i-1),x,v_(i+1),dots,v_k] / det[x_0,dots,v_k]
$

Linear functions on simplicies can be expressed as
$
  u(x) = sum_(i=0)^k lambda^i (x) space u(v_i)
$

= Lagrange Basis
#v(1cm)

If we have a triangulation $mesh$, then the barycentric coordinate functions
can be collected to form the lagrange basis.

We can represent piecewiese-linear (over simplicial cells) functions on the mesh.
$
  u(x) = sum_(i=0)^N b^i (x) space u(v_i)
$


Fullfills Lagrange basis property basis.
$
  b^i (v_j) = delta_(i j)
$
