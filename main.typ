#import "setup.typ": *
#show: notes-template


= Notation

$
  cal(I)^n_k = { I=(i_0,dots,i_(k-1)) quad : quad 0 <= i_i <= n-1 quad forall i }
  \
  hat(cal(I))^n_k = {I=(i_0,dots,i_(k-1)): 0 <= i_0 < i_1 < dots.c < i_(k-1) <= n-1}
$

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

= Mesh and Meshing

This might be an important theoretical result for the thesis:
https://en.wikipedia.org/wiki/Simplicial_approximation_theorem

Barycentric subdivisions is the way to go for refining a simplicial complex.

There is a notion of dual mesh, that is relevant (at least) in DEC.
If we have a simplicial mesh, then the dual mesh is in general not a simplicial mesh.

In Wikipedia this topic is only covered as a dual graph https://en.wikipedia.org/wiki/Dual_graph.

Delaunay traingulations and Voronoi diagrams are dual. Which means constructing a
Voronoi diagram and a delaunay triangulation is equivalent.
This is helpful for FEM mesh generation.

Lloyd's algorithm, a method based on Voronoi diagrams for moving
a set of points on a surface to more evenly spaced positions, is commonly
used as a way to smooth a finite element mesh described by the dual Delaunay
triangulation. This method improves the mesh by making its triangles more
uniformly sized and shaped.

= Numerical Analysis Theory

Fundamental Theorem:
Constistency and Stability => Convergence.

In FEM consistency error, is difference between discretized solution
and exact solution. Includes best approximation error and variational crimes
(changes to bilinear form).

Stability means that the mapping from data to solution must be bounded.

The map from data to solution, is the inverse of the differential operator.
So in other words, this operator must be invertible with an inverse that doesn't
have too big of a norm.

= Mixed Finite Elements

Let's consider a poisson equation.
$
  -Delta u = f 
$

In standard weak form
$
  integral grad dot grad v = integral f v
  quad forall v in H^1
$

Instead we can consider mixed strong formulation:
$
  -div grad u = f
  <==> cases(
    sigma = -grad u,
    div sigma = f,
  )
$

If we transition this to weak:
$
  sigma in H(div), u in L^2
  \
  integral sigma dot tau - integral u div tau = 0
  quad forall tau in H(div)
  \
  integral div sigma v = integral f v
  quad forall v in L^2
$

For 1D:
$

  integral sigma tau - integral u tau' = 0
  quad forall tau in H^1
  \
  integral sigma' v = integral f v
  quad forall v in L^2
$

If we do galerkin discretization with
$P_1 times P_1$ we get a singular system. Failure.

If we take $P_1 times P_0$ success (not in 2D!)

For $P_2 times P_0$ failure by oscillations.

Seldom stable.

== Vector Laplacian

Vector laplacian is just scalar laplacian on each component.
It can also expressed as

$
  curl curl u - grad div u = f
  \
  u dot n = 0 quad curl u times n = 0
$
The conditions $curl u times n = 0$ is actually two equations.

In the weak form
$
  integral (curl u dot curl v + div u div v) = integral f dot v
  quad forall v
$

On a L-shaped domain with a reentrant corner:
Lagrange FE converges, but to the wrong solution!

We have positive definite bilinear form, giving us stability.
But here we have a consistency problem. The consistency error
doesn't go to zero as we refine, because of the strong singularity in $u$.

This doesn't work for any conforming FE!

== Vector Laplacian with non trivial topology

Normally when solving a vector poisson problem with
zero boundary condition and zero right hand side, then
we should get a solution that is also zero.

But!  If there is hole in the domain, like in an annulus,
then this isn't true. The kernel of the vector laplacian is non-trivial!
The dimension of the kernel of the laplacian (space of harmonic forms) is
actually the number of holes (1st betty number).
For this reason the PDE problem isn't even well-posed.
We need to mod out (modulo) the harmonic forms out of $f$
and require that the solution $u$ is orthogonal to all harmonic forms (side condition).

== Maxwell eigenvalue problem

$
  curl curl = lambda u \
  div u = 0
$

In weak form
$
  u in H(curl) quad
  integral curl u dot curl v = lambda integral u dot v
  quad forall v in H(curl)
$


= Discrete calculus

Discrete Calculus is the kind of calculus one needs for finite difference schemes
and for DEC.

https://en.wikipedia.org/wiki/Discrete_calculus

= Homological Algebra

Chain Complex: Sequence of vector spaces and linear maps

$
  dots.c -> V_(k+1) ->^(diff_(k+1)) V_k ->^(diff_k) V_(k-1) -> dots.c
  quad "with" quad
  diff_k compose diff_(k+1) = 0
$

Graded vector space $V = plus.big.circle_k V_k$ with graded linear operator $diff = plus.big.circle_k diff_k$ of degree -1,
such that $diff compose diff = 0$.

$V_k$: $k$-chains \
$diff_k$: $k$-th boundary operator \
$frak(Z)_k = ker diff_k$: $k$-cycles \
$frak(B)_k = im diff_(k+1)$: $k$-boundaries \
$frak(H)_k = frak(Z)_k \/ frak(B)_k$: $k$-th homology space \

== Simplicial Complex

$cal(S)$ simplicial complex.

The span (formal linear combinations) of all simplicies, gives us a vector space.
Together with the boundary operator, this forms a simplicial chain complex.

= Exterior Algebra

We have an vector space $V$ or a field $KK$.\
We first define the tensor algebra
$
  T(V) = plus.circle.big_(k=0)^oo V^(times.circle k)
  = K plus.circle V plus.circle (V times.circle V) plus.circle dots.c
$

Now we define the two-sides ideal $I = { x times.circle x : x in V }$.\
The exterior algebra is now the quotient algebra of the tensor algebra by the ideal
$
  wedgespace(V) = T(V)\/I
  = wedgespace^0(V) plus.circle wedgespace^1(V) plus.circle dots.c plus.circle wedgespace^n (V)
$
The exterior product $wedge$ of two element in $wedgespace(V)$ is then
$
  alpha wedge beta = alpha times.circle beta quad (mod I)
$

We have dimensionality given by the binomial coefficent.
$
  dim wedgespace^k (V) = binom(n,k)
$

The $k$-th exterior algebra $wedgespace^k V$ over the vector space $V$ is
called the space of $k$-vectors.\
The $k$-th exterior algebra $wedgespace^k (V^*)$ over the dual space $V^*$ of $V$ is
called the space of $k$-forms.\


We can easily switch between $k$-vectors and $k$-forms, if we identify the basis of $V$
with the dual(!) basis of $V^*$. We just reinterpret the components.\
This defines two unary operators.
- Flat #flat to move from $k$-vector to $k$-form.
- Sharp #sharp to move from $k$-form to $k$-vector.
This is inspired by musical notation. It moves the tensor index down (#flat) and up (#sharp).

= Differential Geometry

If we have an immersion (maybe also embedding) $f: M -> RR^n$,
then it's differential $dif f: T_p M -> T_p RR^n$, is called the push-forward
and tells us how our intrinsic tangential vectors are being stretched when viewed geometrically.

Computationally this differential $dif f$ can be represented, since it is a linear map, by
a Jacobi Matrix $J$.

The differential tells us also how to take an inner product of our tangent
vectors, by inducing a metric
$
  g(u, v) = dif f(u) dot dif f(v)
$

Computationally this metric is represented, since it's a bilinear form, by a
metric tensor $G$.\
The above relationship then becomes
$
  G = J^transp J
$


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

An arbitrary differential form can be written as (with Einstein sum convention)
$
  alpha = 1/k!
  alpha_(i_1 dots i_k) dif x^(i_1) wedge dots.c wedge dif x^(i_k)
  = sum_(i_1 < dots < i_k) 
  alpha_(i_1 dots i_k) dif x^(i_1) wedge dots.c wedge dif x^(i_k)
$

#pagebreak()
= Exterior Calculus in 3D (vs Vector Calculus)

You can think of $k$-vector field as a *density* of infinitesimal oriented $k$-dimensional.

The differential $k$-form is just a $k$-form field, which is the dual measuring object.

== Derivatives


- $dif_0$: Measures how much a 0-form (scalar field) changes linearly,
  producing a 1-form (line field).
- $dif_1$: Measures how much a 1-form (line field) circulates areally,
  producing a 2-form (areal field).
- $dif_2$: Measures how much a 2-form (areal flux field) diverges volumetrically,
  producing a 3-form (volume field).
  
$
  grad &=^~ dif_0
  quad quad
  &&grad f = (dif f)^sharp
  \
  curl &=^~ dif_1
  quad quad
  &&curl avec(F) = (hodge dif avec(F)^flat)^sharp
  \
  div &=^~ dif_2
  quad quad
  &&"div" avec(F) = hodge dif hodge avec(F)^flat
$

Laplacian
$
  Delta f = div grad f = hodge dif hodge dif f
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


$
  &H    (grad; Omega) &&=^~ H Lambda^0 (Omega) \
  &Hvec (curl; Omega) &&=^~ H Lambda^1 (Omega) \
  &Hvec (div ; Omega) &&=^~ H Lambda^2 (Omega) \
$


$
  0 -> H (grad; Omega) limits(->)^grad Hvec (curl; Omega) limits(->)^curl Hvec (div; Omega) limits(->)^div L^2(Omega) -> 0
  \
  curl compose grad = 0
  quad quad
  div compose curl = 0
$

== FE Spaces

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
  cal(W) Lambda^0 (mesh) &=^~ cal(S)^0_1 (mesh) \
  cal(W) Lambda^1 (mesh) &=^~ bold(cal(N)) (mesh) \
  cal(W) Lambda^2 (mesh) &=^~ bold(cal(R T)) (mesh) \
$


$
  0 -> cal(S)^0_1 (mesh) limits(->)^grad bold(cal(N)) (mesh) limits(->)^curl bold(cal(R T)) (mesh) limits(->)^div cal(S)^(-1)_0 (mesh) -> 0
$


#pagebreak()
= Main Definition and Theorems of Exterior Calculus

== Exact vs Closed

A fundamental fact about exterior differentiation is that $dif(dif omega) = 0$
for any sufficiently smooth differential form $omega$.

Under some restrictions on the topology of $Omega$ the converse is
also true, which is called the exact sequence property:

== Poincaré's lemma
For a contractible domain $Omega subset.eq RR^n$ every
$omega in Lambda^l_1 (Omega), l >= 1$, with $dif omega = 0$ is the exterior
derivative of an ($l − 1$)–form over $Omega$.

== Stokes' Theorem
$
  integral_Omega dif omega = integral_(diff Omega) trace omega
$
for all $omega in Lambda^l_1 (Omega)$


== Leibniz Product rule
$
  dif (alpha wedge beta) = dif alpha wedge beta + (-1)^abs(alpha) alpha wedge dif beta
$

Using the Leibniz Rule we can derive what the exterior derivative of a 1-form
term $alpha_j dif x^j$ must be, if we interpret this term as a wedge $alpha_j
wedge dif x^j$ between a 0-form $alpha_j$ and a 1-form $dif x^j$.
$
  dif (alpha_j dif x^j)
  = dif (alpha_j wedge dif x^j)
  = (dif alpha_j) wedge dif x^j + alpha_j wedge (dif dif x^j)
  = (diff alpha_j)/(diff x^i) dif x^i wedge dif x^j
$

== Integration by parts
$
  integral_Omega dif omega wedge eta
  + (-1)^l integral_Omega omega wedge dif eta
  = integral_(diff Omega) omega wedge eta
$
for $omega in Lambda^l (Omega), eta in Lambda^k (Omega), 0 <= l, k < n − 1, l + k = n − 1$.


$
  integral_Omega dif omega wedge eta
  =
  (-1)^(k-1)
  integral_Omega omega wedge dif eta
  +
  integral_(diff Omega) "Tr" omega wedge "Tr" eta
$

$
  inner(dif omega, eta) = inner(omega, delta eta) + integral_(diff Omega) "Tr" omega wedge "Tr" hodge eta
$


#pagebreak()

= Exterior Derivative
Purely topological, no geometry.

In discrete settings defined as coboundary operator, through Stokes' theorem.\
So the discrete exterior derivative is just the transpose of the boundary operator / incidence matrix.

The exterior derivative is closed in the space of Whitney forms, because of the de Rham complex.

George de Rham isch Schwizer!!!

The local (on a single cell) exterior derivative is always the same for any cell.
Therefore we can compute it on the reference cell.


= Hodge Star operator

The Hodge star operator is a linear operator
$
  hodge: Lambda^k (Omega) -> Lambda^(n-k) (Omega)
$
s.t.
$
  alpha wedge (hodge beta) = inner(alpha, beta)_(Lambda^k) vol
  quad forall alpha in Lambda^k (Omega)
$
where $inner(alpha, beta)$ is the pointwise inner product on #strike[differential] $k$-forms
meaning it's a scalar function on $Omega$.\
$vol = sqrt(abs(g)) dif x^1 dots dif x^n$ is the volume form (top-level form $k=n$).

Given a basis for $Lambda^k (Omega)$, we can get an LSE by replacing $alpha$ with each basis element.\
This allows us to solve for $hodge beta$.\
For a inner product on an orthonormal basis on euclidean space, the solution is explicit and doesn't involve solving an LSE.

In general:\
- $hodge 1 = vol$
- $hodge vol = 1$

Integrating the equation over $Omega$ we get
$
  integral_Omega alpha wedge (hodge beta)
  = integral_Omega inner(alpha, beta)_(Lambda^k) vol
  = inner(alpha, beta)_(L^2 Lambda^k)
  quad forall alpha in Lambda^k (Omega)
$

This equation can be used to find the discretized weak Hodge star operator.\
The weak variational form of our hodge star operator is the mass bilinear form
$
  m(u, v) = integral_Omega hodge u wedge v
$
After Galerkin discretization we get the mass matrix for our discretized weak Hodge star operator
as the $L^2$-inner product on differential $k$-forms.
$
  amat(M)_(i j) = integral_Omega phi_j wedge hodge phi_i = inner(phi_j, phi_i)_(L^2 Lambda^k)
$

This is called the mass bilinear form / matrix, since for 0-forms, it really coincides with
the mass matrix from Lagrangian FEM.

The Hodge star operator captures geometry of problem through this inner product,
which depends on Riemannian metric.\
Let's see what this dependence looks like.

An inner product on a vector space can be represented as a Gramian matrix given a basis.

The Riemannian metric (tensor) is an inner product on the tangent vectors with
basis $diff/(diff x^i)$.\
The inverse metric is an inner product on covectors / 1-forms with basis $dif x^i$.\
This can be further extended to an inner product on #strike[differential] $k$-forms
with basis $dif x_i_1 wedge dots wedge dif x_i_k$.
$
  inner(dif x_I, dif x_J) = det [inner(dif x_I_i, dif x_I_j)]_(i,j)^k
$
Lastly we can extend this to an inner product on differential $k$-forms.\
For (1st order) FEEC we have piecewise-linear differential $k$-forms with the
Whitney basis $lambda_sigma$.\
Therefore our discretized weak hodge star operator is the mass matrix, which is the Gramian matrix
on all Whitney $k$-forms.

$
  amat(M)^k = [inner(lambda_sigma_j, lambda_sigma_i)_(L^2 Lambda^k)]_(0 <= i,j < binom(n,k))
  = [inner(lambda_I, lambda_J)_(L^2 Lambda^k)]_(I,J in hat(cal(I))^n_k)
$


== Computation of Hodge Star in Tensor Index Notation

Attention Einstein sum convention ahead!

This is the formula for the hodge star of basis k-forms.
$
  hodge (dif x^(i_1) wedge dots.c wedge dif x^(i_k))
  = sqrt(abs(det[g_(a b)])) / (n-k)! g^(i_1 j_1) dots.c g^(i_k j_k)
  epsilon_(j_1 dots j_n) dif x^(j_(k+1)) wedge dots.c wedge dif x^(j_n)
$

Here with restricted to increasing indices $j_(k+1) < dots < j_n$
$
  hodge (dif x^(i_1) wedge dots.c wedge dif x^(i_k))
  = sqrt(abs(det[g_(a b)])) sum_(j_(k+1) < dots < j_n)
  g^(i_1 j_1) dots.c g^(i_k j_k)
  epsilon_(j_1 dots j_n) dif x^(j_(k+1)) wedge dots.c wedge dif x^(j_n)
$

For an arbitrary differential k-form $alpha$, we have
$
  hodge alpha = sum_(j_(k+1) < dots < j_n)
  (hodge alpha)_(j_(k+1) dots j_n) dif x^(j_(k+1)) wedge dots.c wedge dif x^(j_n)
$

$
  (hodge alpha)_(j_(k+1) dots j_n)
  = sqrt(det[g_(a b)]) / k!
  alpha_(i_1 dots i_k) g^(i_1 j_1) dots.c g^(i_k j_k) epsilon_(j_1 dots j_n)
$


== DEC considerations

In DEC the discrete Hodge-star $hat(hodge)$ is defined using a dual mesh.
In DEC we integrate the continuous $k$-forms over the $k$-simplicies of the mesh,
to obtain a $k$-cochain, a function assigning a number to each $k$-simplex in the mesh.

When doing this discretization, we only integrate along $k$ dimensions and
only capture this information. The remaining $n-k$ dimensions are ignored.
But the Hodge dual of a $k$-form lives in exactly this $n-k$ dual space!
So for the discretization we need next to the primal mesh, also a dual mesh,
consisting of dual cells (not necessarily simplicies).

The diagonal discrete hodge star is then defined as
$
  hat(hodge) hat(alpha)_j = vol(sigma^star_j) / vol(sigma_j) hat(alpha)_j
$
This reminds me a lot of the lumped mass matrix, obtained when approximating
the mass bilinear form ($L^2$-inner product) using the trapezoidal quadrature rule on 0-forms.
This could really be the case, since the weak form of the hodge star operator is really
the generalized mass bilinear form ($L^2 Lambda^k$-inner product) \
I should check if the this diagonal hodge star is a true generalization of the lumped mass matrix.

DEC	can	be obtained	from the lowest	order	FEEC by	replacing
the mass matrices	by ones	that are computed	from primal-dual meshes.

One distinguishing aspect of DEC is this geometric nature of the mass matrix.
The dual meshes are circumcentric duals and in the early years it was known
that simplices which contain their own circumcenters are sufficient for
producing appropriate mass matrices. In recent work we proved that the much
broader class of Delaunay meshes (modulo some boundary restrictions) suffice.


= $L^2$-inner product

The inner product is defined as
$
  inner(omega, eta)_(L^2 Lambda^k)
  =
  integral_Omega inner(omega_x, eta_x) "vol"
  =
  integral omega wedge hodge eta
$


= Coderivative Operator

Coderivative operator $delta: Lambda^k (Omega) -> Lambda^(k-1) (Omega)$
defined such that
$
  hodge delta omega = (-1)^k dif hodge omega
  \
  delta_k := (dif_(k-1))^* = (-1)^k space (hodge_(k-1))^(-1) compose dif_(n-k) compose hodge_k
$

For vanishing boundary it's the formal $L^2$-adjoint of the exterior derivative.


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



#pagebreak()
= Hodge-Laplace Source Problem

== Primal Strong form
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


== Primal Weak Form

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
  integral_Omega ((dif delta + delta dif) u) wedge hodge v = integral_Omega f wedge hodge v
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
  integral_Omega (delta u) wedge hodge (delta v) + integral_Omega (dif u) wedge hodge (dif v) = integral_Omega f wedge hodge v
  quad
  forall v in H Lambda^k (Omega)
$

== Galerkin Primal Weak Form

$
  u_h = sum_(i=1)^N mu_i phi_i
  quad quad
  v_h = phi_j
  \
  u in H Lambda^k (Omega): quad quad
  inner(delta u, delta v) + inner(dif u, dif v) = inner(f, v)
  quad quad forall v in H Lambda^k (Omega)
  \
  vvec(mu) in RR^N: quad
  sum_(i=1)^N mu_i (integral_Omega (delta phi_i) wedge hodge (delta phi_j) + integral_Omega (dif phi_i) wedge hodge (dif phi_j))
  =
  sum_(i=1)^N mu_i integral_Omega f wedge hodge phi_j
  quad forall j in {1,dots,N}
$

$
  amat(A) vvec(mu) = 0
  \
  A =
  [integral_Omega (delta phi_i) wedge hodge (delta phi_j)]_(i,j=1)^N
  +
  [integral_Omega (dif phi_i) wedge hodge (dif phi_j)]_(i,j=1)^N
  \
  vvec(phi) = [integral_Omega f wedge hodge phi_j]_(j=1)^N
$


== Mixed Strong Formulation

Given $f in Lambda^k$, find $(sigma,u,p) in (Lambda^(k-1) times Lambda^k times frak(h)^k)$ s.t.
$
  sigma - delta u &= 0
  quad &&"in" Omega
  \
  dif sigma + delta dif u &= f - p
  quad &&"in" Omega
  \
  tr hodge u &= 0
  quad &&"on" diff Omega
  \
  tr hodge dif u &= 0
  quad &&"on" diff Omega
  \
  u perp frak(h)
$


== Mixed Weak Form

Given $f in L^2 Lambda^k$, find $(sigma,u,p) in (H Lambda^(k-1) times H Lambda^k times frak(h)^k)$ s.t.
$
  inner(sigma,tau) - inner(u,dif tau) &= 0
  quad &&forall tau in H Lambda^(k-1)
  \
  inner(dif sigma,v) + inner(dif u,dif v) + inner(p,v) &= inner(f,v)
  quad &&forall v in H Lambda^k
  \
  inner(u,q) &= 0
  quad &&forall q in frak(h)^k
$

== Galerkin Mixed Weak Form
$
  sum_j sigma_j inner(phi^(k-1)_j,phi^(k-1)_i) - sum_j u_j inner(phi^k_j,dif phi^(k-1)_i) &= 0
  \
  sum_j sigma_j inner(dif phi^(k-1)_j,phi^k_i) + sum_j u_j inner(dif phi^k_j,dif phi^k_i) + sum_j p_j inner(eta^k_j,phi^k_i) &= sum_j f_j inner(psi_j,phi^k_i)
  \
  sum_j u_j inner(phi^k_j,eta^k_i) &= 0
$

$
  hodge sigma - dif^transp hodge u &= 0
  \
  hodge dif sigma + dif^transp hodge dif u + hodge H p &= hodge f
  \
  H^transp hodge u &= 0
$

$
  mat(
    hodge, -dif^transp hodge, 0;
    hodge dif, dif^transp hodge dif, hodge H;
    0, H^transp hodge, 0;
  )
  vec(sigma, u, p)
  =
  vec(0, hodge f, 0)
$

Alternatively we have (from the "DEC vs FEEC" poster)

Given $f$, find $(sigma, u)$ s.t.
$
  mat(
    hodge, -dif^transp hodge;
    hodge dif, dif^transp hodge dif
  )
  vec(sigma, u)
  =
  vec(0, hodge(f - p))
  \
  "subject to" H^transp hodge u = 0
$
where $p$ is the harmonic part of $f$ and $H$ is a basis for discrete harmonics.

#pagebreak()
= Hodge-Laplace Eigenvalue Problem

== Primal Strong Form
$
  (delta dif + dif delta) u = lambda u
$

== Mixed Weak Form

Find $lambda in RR$, $(sigma, u) in (H Lambda^(k-1) times H Lambda^k \\ {0})$, s.t.
$
  inner(sigma, tau) - inner(u, dif tau) &= 0
  quad &&forall tau in H Lambda^(k-1)
  \
  inner(dif sigma, v) + inner(dif u, dif v) &= lambda inner(u,v)
  quad &&forall v in H Lambda^k
$


== Galerkin Mixed Weak Form
$
  sum_j sigma_j inner(phi^(k-1)_j, phi^(k-1)_i) - sum_j u_j inner(phi^k_j, dif phi^(k-1)_i) &= 0
  \
  sum_j sigma_j inner(dif phi^(k-1)_j, phi^k_i) + sum_j u_j inner(dif phi^k_j, dif phi^k_i) &= lambda sum_j u_j inner(phi^k_j,phi^k_i)
$

$
  hodge sigma - dif^transp hodge u &= 0
  \
  hodge dif sigma + dif^transp hodge dif u &= lambda hodge u
$

$
  mat(
    hodge, -dif^transp hodge;
    hodge dif, dif^transp hodge dif;
  )
  vec(sigma, u)
  =
  lambda
  mat(
    0,0;
    0,hodge
  )
  vec(sigma, u)
$

This is a symmetric indefinite sparse generalized matrix eigenvalue problem,
that can be solved by an iterative eigensolver such as Krylov-Schur.
This is also called a GHIEP problem.

#pagebreak()

= Derivation of Analytic Element Matrix Formulas

This is the heart of the FEEC implementation!

== Exterior Derivative of Whitney Forms on Reference Cell

The exterior derivative of a Whitney form:
$
  dif lambda_(i_0 dots i_k)
  &= k! sum_(l=0)^k (-1)^l dif lambda_i_l wedge
  (dif lambda_i_0 wedge dots.c wedge hat(dif lambda_i_l) wedge dots.c wedge dif lambda_i_k)
  \
  &= k! sum_(l=0)^k (-1)^l (-1)^l
  (dif lambda_i_0 wedge dots.c wedge dif lambda_i_l wedge dots.c wedge dif lambda_i_k)
  \
  &= (k+1)! dif lambda_i_0 wedge dots.c wedge dif lambda_i_k
$

Therefore for $i_0 != 0$ we have
$
  dif lambda_(i_0 dots i_k)
  &= (k + 1)! dif x^(i_0-1) wedge dots.c wedge dif x^(i_k-1)
$

And for $i_0 = 0$ we have
$
  dif lambda_(0 i_1 dots i_k)
  &= (k+1)! dif lambda_0 wedge dif lambda_i_1 wedge dots.c wedge dif lambda_i_k
  \
  &= (k+1)! (-sum_(j=0)^(n-1) dif x^j) wedge dif x^(i_1-1) wedge dots.c wedge dif x^(i_k-1)
  \
  &= -(k+1)! sum_(j=0)^(n-1) dif x^j wedge dif x^(i_1-1) wedge dots.c wedge dif x^(i_k-1)

$


$
  L^k = [inner(dif lambda_tau, dif lambda_sigma)_(L^2 Lambda^(k+1) (K))]_(sigma,tau in Delta_k (K))
$

$
  L^0 = [inner(dif lambda_j, dif lambda_i)_(L^2 Lambda^1 (K))]_(i,j in Delta_0 (K))
$


== Mass Bilinear form

$
  M = [inner(lambda_tau, lambda_sigma)_(L^2 Lambda^k (K))]_(sigma,tau in Delta_k (K))
$

$
  inner(lambda_(i_0 dots i_k), lambda_(j_0 dots j_k))_(L^2)
  = k!^2 sum_(l=0)^k sum_(m=0)^k (-)^(l+m) innerlines(
    lambda_i_l (dif lambda_i_0 wedge dots.c wedge hat(dif lambda)_i_l wedge dots.c wedge dif lambda_i_k),
    lambda_j_m (dif lambda_j_0 wedge dots.c wedge hat(dif lambda)_j_m wedge dots.c wedge dif lambda_j_k),
  )_(L^2) \
  = k!^2 sum_(l,m) (-)^(l+m) integral_K innerlines(
    lambda_i_l (dif lambda_i_0 wedge dots.c wedge hat(dif lambda)_i_l wedge dots.c wedge dif lambda_i_k),
    lambda_j_m (dif lambda_j_0 wedge dots.c wedge hat(dif lambda)_j_m wedge dots.c wedge dif lambda_j_k),
  ) vol \
  = k!^2 sum_(l,m) (-)^(l+m) integral_K lambda_i_l lambda_j_m innerlines(
    dif lambda_i_0 wedge dots.c wedge hat(dif lambda)_i_l wedge dots.c wedge dif lambda_i_k,
    dif lambda_j_0 wedge dots.c wedge hat(dif lambda)_j_m wedge dots.c wedge dif lambda_j_k,
  ) vol \
  = k!^2 sum_(l,m) (-)^(l+m) innerlines(
    dif lambda_i_0 wedge dots.c wedge hat(dif lambda)_i_l wedge dots.c wedge dif lambda_i_k,
    dif lambda_j_0 wedge dots.c wedge hat(dif lambda)_j_m wedge dots.c wedge dif lambda_j_k,
  )
  integral_K lambda_i_l lambda_j_m vol \
$

Let's assume increasing indices.

For a reference simplex, we have

If $i_0 != 0$
$
  dif lambda_i_0 wedge dots.c wedge hat(dif lambda)_i_l wedge dots.c wedge dif lambda_i_k
  =
  dif x^(i_0 - 1) wedge dots.c wedge hat(dif x)^(i_l - 1) wedge dots.c wedge dif x^(i_k - 1)
$
else if $i_0 = 0$
$
  dif lambda_i_0 wedge dots.c wedge hat(dif lambda)_i_l wedge dots.c wedge dif lambda_i_k
  &=
  (-sum_(j=0)^(n-1) dif x^j) wedge dif x^(i_1 - 1) wedge dots.c wedge hat(dif x)^(i_l - 1) wedge dots.c wedge dif x^(i_k - 1)
  \ &=
  -sum_(j=0)^(n-1) dif x^j wedge dif x^(i_1 - 1) wedge dots.c wedge hat(dif x)^(i_l - 1) wedge dots.c wedge dif x^(i_k - 1)
$



#pagebreak()
= Barycentric Coordinates

Barycentric coordinates exist for all $k$-simplicies.\
They are a coordinate system relative to the simplex.\
Where the barycenter of the simplex has coordinates $(1/k)^k$.\

$
  x = sum_(i=0)^k lambda^i (x) space v_i
$
with $sum_(i=0)^k lambda_i(x) = 1$ and inside the simplex $lambda_i (x) in [0,1]$ (partition of unity).

$
  lambda^i (x) = det[v_0,dots,v_(i-1),x,v_(i+1),dots,v_k] / det[x_0,dots,v_k]
$

Linear functions on simplicies can be expressed as
$
  u(x) = sum_(i=0)^k lambda^i (x) space u(v_i)
$

The following integral formula for powers of barycentric coordinate functions holds (NUMPDE):
$
  integral_K lambda_0^(alpha_0) dots.c lambda_n^(alpha_n) vol
  =
  n! abs(K) (alpha_0 ! space dots.c space alpha_n !)/(alpha_0 + dots.c + alpha_n + n)!
$
where $K in Delta_n, avec(alpha) in NN^(n+1)$.\
The formula treats all barycoords symmetrically.

For piecewise linear FE, the only relevant results are:
$
  integral_K lambda_i lambda_j vol
  = abs(K)/((n+2)(n+1)) (1 + delta_(i j))
$

$
  integral_K lambda_i vol = abs(K)/(n+1)
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

#pagebreak()
= Whitney Forms and Whitney Basis
#v(1cm)

Whitney $k$-forms $cal(W) Lambda^k (mesh)$ are piecewise-constant (over cells $Delta_k (mesh)$)
differential $k$-forms.

The defining property of the Whitney basis is a from pointwise to integral
generalized Lagrange basis property (from interpolation):\
For any two $k$-simplicies $sigma, tau in Delta_k (mesh)$, we have
$
  integral_sigma lambda_tau = cases(
    +&1 quad &"if" sigma = +tau,
    -&1 quad &"if" sigma = -tau,
     &0 quad &"if" sigma != plus.minus tau,
  )
$

The Whitney $k$-form basis function live on all $k$-simplicies of the mesh $mesh$.
$
  cal(W) Lambda^k (mesh) = "span" {lambda_sigma : sigma in Delta_k (mesh)}
$

There is a isomorphism between Whitney $k$-forms and cochains.\
Represented through the de Rham map (discretization) and Whitney interpolation:\
- The integration of each Whitney $k$-form over its associated $k$-simplex yields a $k$-cochain.
- The interpolation of a $k$-cochain yields a Whitney $k$-form.\


Whitney forms are affine invariant. \
Let $sigma = [x_0 dots x_n]$ and $tau = [y_0 dots y_n]$ and $phi: sigma -> tau$
affine map, such that $phi(x_i) = y_i$, then
$
  cal(W)[x_0 dots x_n] = phi^* (cal(W)[y_0 dots y_n])
$

The Whitney basis ${lambda_sigma}$ is constructed from barycentric coordinate functions ${lambda_i}$.

$
  lambda_(i_0 dots i_k) =
  k! sum_(l=0)^k (-1)^l lambda_i_l
  (dif lambda_i_0 wedge dots.c wedge hat(dif lambda_i_l) wedge dots.c wedge dif lambda_i_k)
$

Some expansions:
#align(center)[#grid(
  columns: 2,
  gutter: 10%,
  $
    cal(W)[v_0 v_1] =
    &-lambda_1 dif lambda_0
     + lambda_0 dif lambda_1 
    \
    cal(W)[v_0 v_1 v_2] =
    &+2 lambda_2 (dif lambda_0 wedge dif lambda_1) \
    &-2 lambda_1 (dif lambda_0 wedge dif lambda_2) \
    &+2 lambda_0 (dif lambda_1 wedge dif lambda_2) \
  $,
  $
    cal(W)[v_0 v_1 v_2 v_3] =
    - &6 lambda_3 (dif lambda_0 wedge dif lambda_1 wedge dif lambda_2) \
    + &6 lambda_2 (dif lambda_0 wedge dif lambda_1 wedge dif lambda_3) \
    - &6 lambda_1 (dif lambda_0 wedge dif lambda_2 wedge dif lambda_3) \
      &6 lambda_0 (dif lambda_1 wedge dif lambda_2 wedge dif lambda_3) \
  $
)]

#pagebreak()

== Reference 1-simplex

$
  v_0 = (0), v_1 = (1)
  \
  e = [v_0 v_1]
$

Barycentric Coordinate Functions
$
  &lambda_0 = 1 - x
  quad quad
  &&dif lambda_0 = -dif x
  \
  &lambda_1 = x
  quad quad
  &&dif lambda_1 = +dif x
$

$
  cal(W)[e] = cal(W)[v_0 v_1] = dif x
$

$
  hodge lambda_0 = 1-x dif x
  \
  hodge lambda_1 = x dif x
  \
  \
  hodge lambda_(0 1) = 1
$

$
  diff_1 = mat(
    ,e;
    v_0,-1;
    v_1,+1;
    augment: #(hline: 1, vline: 1)
  )
  \
  dif_0 = mat(
    ,lambda_0,lambda_1;
    lambda_e,-1, +1;
    augment: #(hline: 1, vline: 1)
  )
$

$
  dif_0 cal(W)[v_0] = mat(-1,+1) mat(1;0) = [-1] = -cal(W)[e] = -dif x quad checkmark
  \
  dif_0 cal(W)[v_1] = mat(-1,+1) mat(0;1) = [+1] = +cal(W)[e] = +dif x quad checkmark
$

== Reference 2-simplex

$
  v_0=(0,0),v_1=(1,0),v_2=(0,1)
  \
  e_0=[v_0 v_1],e_1=[v_0 v_2],e_2=[v_1 v_2]
  \
  K=[v_0 v_1 v_2]
$

Barycentric Coordinate Functions
$
  &lambda_0 = 1 - x_0 - x_1
  quad quad
  &&dif lambda_0 = -dif x_0 -dif x_1
  \
  &lambda_1 = x_0
  quad quad
  &&dif lambda_1 = +dif x_0
  \
  &lambda_2 = x_1
  quad quad
  &&dif lambda_2 = +dif x_1
$


$
  &cal(W)[e_0] = cal(W)[v_0 v_1] = (1-x_1) dif x_0 + x_0 dif x_1
  quad quad
  &&dif cal(W)[e_0] = +2 dif x_0 wedge dif x_1
  \
  &cal(W)[e_1] = cal(W)[v_0 v_2] = x_1 dif x_0 + (1-x_0) dif x_1
  quad quad
  &&dif cal(W)[e_1] = -2 dif x_0 wedge dif x_1
  \
  &cal(W)[e_2] = cal(W)[v_1 v_2] = -x_1 dif x_0 + x_0 dif x_1
  quad quad
  &&dif cal(W)[e_2] = +2 dif x_0 wedge dif x_1
  \
  \
  &cal(W)[K] = cal(W)[v_0 v_1 v_2] = 2 dif x_0 wedge dif x_1
$

$
  hodge lambda_0 = (1 - x - y) dif x wedge dif y
  \
  hodge lambda_1 = x dif x wedge dif y
  \
  hodge lambda_2 = y dif x wedge dif y
  \
  \
  hodge lambda_(0 1) = -x dif x + (1-y) dif y
  \
  hodge lambda_(0 2) = (x-1) dif x + y dif y
  \
  hodge lambda_(1 2) = x dif x - y dif y
  \
  \
  hodge lambda_(0 1 2) = 2
$

$
  diff_2 = mat(
    ,k;
    e_0,+1;
    e_1,-1;
    e_2,+1;
    augment: #(hline: 1, vline: 1)
  )
  quad quad
  diff_1 = mat(
    ,e_0,e_1,e_2;
    v_0,-1,-1, 0;
    v_1,+1, 0,-1;
    v_2, 0,+1,+1;
    augment: #(hline: 1, vline: 1)
  )
  \
  dif_1 = mat(
    ,lambda_e_0,lambda_e_1,lambda_e_2;
    lambda_K,+1,-1,+1;
    augment: #(hline: 1, vline: 1)
  )
  quad quad
  dif_0 = mat(
    ,lambda_0,lambda_1,lambda_2;
    lambda_e_0,-1,+1, 0;
    lambda_e_1,-1, 0,+1;
    lambda_e_2, 0,-1,+1;
    augment: #(hline: 1, vline: 1)
  )
$

= Reference 3-simplex

$
  v_0=(0,0,0),v_1=(1,0,0),v_2=(0,1,0),v_3=(0,0,1)
  \
  e_0=[v_0 v_1],e_1=[v_0 v_2],e_2=[v_0 v_3],e_3=[v_1 v_2],e_4=[v_1 v_3],e_5=[v_2 v_3]
  \
  K_0=[v_0 v_1 v_2],K_1=[v_0 v_1 v_3],K_2=[v_0 v_2 v_3],K_3=[v_1 v_2 v_3]
$

Barycentric Coordinate Functions
$
  &lambda_0 = 1 - x_0 - x_1 - x_2
  quad quad
  &&dif lambda_0 = -dif x_0 -dif x_1 -dif x_2
  \
  &lambda_1 = x_0
  quad quad
  &&dif lambda_1 = +dif x_0
  \
  &lambda_2 = x_1
  quad quad
  &&dif lambda_2 = +dif x_1
  \
  &lambda_3 = x_2
  quad quad
  &&dif lambda_3 = +dif x_2
$


$
  &cal(W)[e_0] = cal(W)[v_0 v_1] = (1-x_1-x_2) dif x_0 + x_0 dif x_1 + x_0 dif x_2
  quad quad
  &&dif cal(W)[e_0] = +2 dif x_0 wedge dif x_1 + 2 dif x_0 wedge dif x_2
  \
  &cal(W)[e_1] = cal(W)[v_0 v_2] = x_1 dif x_0 + (1-x_0-x_2) dif x_1 + x_1 dif x_2
  quad quad
  &&dif cal(W)[e_1] = -2 dif x_0 wedge dif x_1 + 2 dif x_1 wedge dif x_2
  \
  &cal(W)[e_2] = cal(W)[v_0 v_3] = x_2 dif x_0 + x_2 dif x_1 + (1-x_0-x_1) dif x_2
  quad quad
  &&dif cal(W)[e_2] = -2 dif x_0 wedge dif x_2 - 2 dif x_1 wedge dif x_2
  \
  &cal(W)[e_3] = cal(W)[v_1 v_2] = -x_1 dif x_0 + x_0 dif x_1
  quad quad
  &&dif cal(W)[e_3] = 2 dif x_0 wedge dif x_1
  \
  &cal(W)[e_4] = cal(W)[v_1 v_3] = -x_2 dif x_0 + x_0 dif x_2
  quad quad
  &&dif cal(W)[e_4] = 2 dif x_0 wedge dif x_2
  \
  &cal(W)[e_5] = cal(W)[v_2 v_3] = -x_2 dif x_1 + x_1 dif x_2
  quad quad
  &&dif cal(W)[e_5] = 2 dif x_1 wedge dif x_2
$

$
  &cal(W)[K_0] = cal(W)[v_0 v_1 v_2] = (2-2x_2) dif x_0 wedge dif x_1 + 2 x_1 dif x_0 wedge dif x_2 - 2 x_0 dif x_1 wedge dif x_2
$



#pagebreak()
= The Laplacian

We define the Laplacian here as the negative of the usual laplacian!

The weak Laplacian is a an operator $Delta: H^2(Omega) -> L^2(Omega)$.

It is is self-adjoint
$
  inner(Delta u, v) = inner(u, Delta v)
$

It is positive-semidefinite!
$
  inner(Delta phi, phi) >= 0
$


The Poisson Problem is
$
  Delta phi = rho
$

FEM and DEC give rise to the same cotan formula in 2D.

The Laplace-Beltrami opertor is the ordinary Laplace operator in Differential
Geometry on curved surfaces.

A twice-differentiable function $phi: Omega -> RR$ is called harmonic if it is
in the kernel of the Laplacian, i.e, $Delta phi = 0$.

The only harmonic function on a compact connected domain without boundary are
the constant functions. This follows from the strong maximum principle.

This implies we can add a constant to any solution of a Poisson equation and it
will still be a solution. This follows from Gradinaru's "Fundamental Theorem of Linear Algebra".
Since the Laplacian is self-adjoint, it's image is the orthogonal complement of its kernel,
which consists of constant functions. Therefore $inner(Delta phi, c) = 0$, and therefore
$c in.not im(Delta)$.

On a compact domain without boundary, constant functions are not in the image of the Laplacian,
i.e., there is no function $phi$ such that $Delta phi = c$.

Therefore if $rho$ has a constant component then our Poisson problem is not well-posed.\
Sometimes a solution is to solve a modified problem where we removed the constant component.
$Delta phi = rho - macron(rho)$ with $macron(rho) = 1/vol(Omega) integral_Omega rho vol$.

We have Green's first identity
$
  inner(Delta f, g) = -inner(grad f, grad g) + inner(hat(n) dot grad f, g)_(diff)
$


= Maxwell's Equations

Relativistic Electrodynamics
- Maxwell's Equations on 4D Spacetime Manifold!
- Faraday 2-form $F = E wedge dif t + B$
- Current 3-form $J = rho + J wedge dif t$
$
  dif F = 0 \
  dif (hodge F) = J
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


#pagebreak()
= Geometry Processing

== The smoothest function

The solution of the Laplace Equation is the most smooth function there is. Why?
Well the solution of the Laplace equation, minimizes the Dirichlet energy
$
  J(u) = 1/2 integral_Omega norm(grad u)^2 vol
$
Meaning it's the function for which the gradiant changes the least! The smoothest in a sense.\
The Laplacian is the derivative of it
$
  Dif J(u) = Delta u
  quad quad
  Dif J(u)[v] = Delta u v
$

== Smoothing manifolds and Implicit mean curvature flow

If we have a parametrization $phi: M -> RR^d$, then the Dirichlet energy is manifold volume.
E.g. For a curve in $RR^2$ it is arc length.
So if we minimize this, we will obtain the a smoothed out manifold. A minimal surface!!!
$
  J(phi) = integral_Omega norm(phi_1)^2 + dots.c + norm(phi_d)^2 vol_d
$

If we apply the heat equation instead, we will slowly converge to the minimal surface
and can view how it smooths out! Mean curvature flow!!!

#pagebreak()
= D. Arnold Lecture

== Homology

Chains $V_k$ \
Cycles $frak(Z) = cal(N) (V_k)$ \
Boundaries $frak(B) = cal(R) (V_k)$ \
Cohomology $frak(H) = frak(Z)\/frak(B)$

Cycles modulo Boundaries is Cohomology.
So the cycles that are not boundaries of something.
These are holes.

If we have a chain complex, by dualizing it we can obtain a cochain complex.
It contains the same information and is isomorphic.
An example of this is the simplicial chain complex and it's dual the simplicial
cochain complex. So the linear forms on simplicies.

The de Rham map is a isomorphism on cohomology between
the de Rham complex (differential structure) and the simplicial cochain complex
which is once again isomorphic to the simplicial chain complex (topological structure).

This gives us a way of measuring holes of a domain using differentials/PDEs.

A great example is the annulus domain. Here we can consider the funciton
If we work with polar coodinates, then $grad theta$ is a well defined function,
it's single valued everywhere. But there is no $C^oo$ function $theta$ since
it's either multivalued of discontinuous when going around the annulus.
But it has zero curl, since $curl grad theta = 0$. Meaning it's a cycle,
but not a boundary. It's a cohomolgy element. It's a hole!
This span of this function is the whole cohomology space. Therefore it has
dimension one and this tells us there is just one hole.

== Unbounded Operators on Hilbert spaces

Always a combination of a operator together with the space it's defined on
(important!). The space it's defined on differs from it's domain.

Graph and Graph norm.

A closed operator is a weaker form a bounded operator, that shares a lot
of it's properties. A operator is closed if it is topologically closed in it's graph norm.

== Hilbert Complexes (de Rham)

The adjoint of the differential operators, are the counterparts
in the integration by parts formulas.

An adjoint on a domain without boundary conditions, lives on a dual space
with boudary condtions and vice versa.

The harmonic forms are designated representatives of the cohomology space.
So instead of working with quotient spaces, we can just work with harmonic forms.

Harmonic forms are defined as
$
  frak(h)^k = { u in V^k sect V_k^star : dif u = 0, dif^star u = 0 }
$

=== Hodge Laplacian

The abstract Hodge Laplacian is defined on any Hilbert complex.
If it is the de Rham complex, then it is the normal Hodge Laplacian.

The hilbert complex is:
$
  W^(k-1) arrows.rl^dif_(dif^star) W^k arrows.rl^dif_(dif^star) W^(k+1)
$

The abstract hodge laplacian is defined as
$
  L := dif^star dif + dif dif^star
$

It's domain is
$
  D(L^k) = { u in V^k sect V_k^star : dif u in V_(k+1)^star, d^star u in V^(k-1) }
$

The null space of the laplacian are really the harmonic forms (name justified)!
$
  cal(N)(L^k) = frak(h)^k, quad frak(h)^k perp cal(R)(L^k)
$

In 3D we have the following domain(!) complexes:
$
  0 -> H^1 ->^grad H(curl) ->^curl H(div) ->^div L^2 -> 0
  \
  0 <- L^2 <-^(-div) H0(div) <-^curl H0(curl) <-^(-grad) H0^1 <- 0
$
The full spaces $W$ are always $L^2$.


The Hodge-Laplace problem is\
Given $f in W$, find $u in D(L)$ s.t.
$
  L u = f, quad u perp cal(h)
$
The side conditions is important, because the harmonic forms $cal(h)$ are
the null space of the laplacian $L$, and if the null space is non trivial,
then the laplacian operator is not regular, meaning it doesn't have
unique solutions. So it's not well posed. So it's necessary to restrict our problem
further to get uniqueness. The orthogonality to the kernel is necessary.

For a standard scalar homogenuous neumann laplace problem this conditions is the
vanishing mean condition $integral u = 0$.

But the problem is still not well-posed.
If we take the inner product of the range of $L$ with a harmonic form $p in frak(h)$,
we get $(L u, p) = (dif^star u, dif^star p) + (dif u, dif p) = 0$, meaning
$p perp cal(R)(L)$. This is acutally not suprising, since $L$ is self-adjoint
and then the range is orthogonal to the kernel.

For this reason our problem is not well posed, because we can't find $u$ for any $f$.
Since the harmonic forms are for instance not in the image of $L$!
So we need to get the unharmonic part of $f$. Only then it works!
We split $W = frak(h) perp.circle frak(f)^perp$ and $f = P_frak(h) f + (f - P_frak(h) f)$
So instead we modify such that
$
  L u &= f - P_frak(h) f \
  &= f quad (mod frak(h))
$

For a standard scalar neumann problem this is actually
$-Delta u = f - macron(f)$, where $macron(f)$ is the average of $f$.
It's a projection into constants.

Our well-posed strong formulation is\
Given $f in W$, find $u in D(L)$ s.t.
$
  L u = f - P_frak(h) f quad u perp cal(h)
$

The primal weak formulation is\
Find $u in V sect V^star sect frak(h)^perp$, s.t.
$
  inner(dif u, dif v) + inner(dif^star u , dif^star v) = inner(f, v)
  quad forall v in V sect V^star sect frak(h)^perp
$
This is a weaker form, since only $u$ needs to be in the domain of $dif$ and $dif^star$,
but these images don't need to be in the domain of the other operator (as it's
the case for the strong laplacian).
The projection of $f$ is already taken care of, by the restriction of
test function to harmonic forms.

For our standard Neumann problem, we just get\
Find $u in H^1 sect RR^perp$ s.t.
$
  inner(grad u, grad v) = inner(f, v)
  quad forall v in H^1 sect RR^perp
$
So we ignore the constant part of each function.
Make the mean vanish!

For the mixed weak formulation we introduce
new variables $sigma = dif^star u$ and $p = P_frak(h) f$\
Find $(sigma, u, p) in (V^(k-1) times V^k times frak(h)^k)$ s.t.
$
  inner(sigma, tau) - inner(u, dif tau) = 0
  quad forall tau in V^(k-1)  
  \
  inner(dif sigma, v) + inner(dif u, dif v) + inner(p, v) = inner(f, v)
  quad forall v in V^k
  \
  inner(u, q) = 0
  quad forall q in frak(h)^k
$

The first equation is just the definition of the variable $sigma = d^star u$.\
The second equation says $dif sigma + dif^star dif u = f - p$ and inserting $sigma$ we get
$dif dif^star u + dif^star dif u = f - p$.

For the 0-form laplacian, there is no $sigma$ and $tau$, only the primal
weak form is left.

Suprising result! All three formulations are EQUIVALENT!!!
And all of them are well posed: There exists a unique solution.

*Two key properties of all closed Hilbert complexes:*
- Hodge Decomposition
- Poincaré inequality

There exists a _Hodge decomposition_.

We have the following split for each hilbert space
$
  W = frak(Z) perp.circle frak(Z)^perp
$
furthermore we have the following split for cycles.
$
  frak(Z) = frak(B) perp.circle frak(h)
$

*Theorem (Hodge decomposition):*
$
  W = underbrace(frak(B) perp.circle frak(h), frak(Z)) perp.circle underbrace(frak(B)^*, frak(Z)^perp)
$

$
  V = underbrace(frak(B) perp.circle frak(h), frak(Z)) perp.circle frak(Z)^(perp V)
$


*Theorem (Poincaré inequality):*
$
  norm(v)_V <= c_p norm(dif v)
  quad forall v in frak(Z)^(perp V)
$

#pagebreak()
=== Hodge Laplacian in 3D

#table(
  columns: 6,
  align: center,
  //stroke: (x, y) => if y == 0 {(bottom: fgcolor)},
  stroke: fgcolor,
  table.header($k$, $L_k = dif^star dif + dif dif^star$, $tilde(L)_k = inner(dif, dif) + inner(dif^star, dif^star)$, [natural BC], [essential BC], $V^(k-1) times V^k$),
  $0$, $-div grad + 0$, $inner(grad, grad) + 0$, $diff u\/diff n$, [-], $H^1$,
  $1$, $curl curl - grad div$, $inner(curl, curl) + inner(div, div)$, $curl u times n$, $u dot n$, $H^1 times H(curl)$,
  $2$, $-grad div + curl curl$, $inner(div, div) + inner(curl, curl)$, $div u$, $u times n$, $H(curl) times H(div)$,
  $3$, $0 -div grad$, $0 + inner(grad, grad)$, [-], $u$, $H(div) times L^2$,
)
The problems are the same mirrored about the horizontal just with different
boudnary conditions. \
The 0-form and n-form problems are scalar Laplacians with Neumann and Dirichlet b.c. respectively. \
The 1-form and 2-form problems are vector Laplacians with magnetic and ? b.c. respectively. \

The domain of $d^*$ is always $H0(d^*)$, meaning it is a space that consists of forms that are zero on the boundary.
From this we get all boundary condtions of our problems.
The primal natural b.c. come from the term $d^star d$, meaning the derivative of the function needs to be zero on the boundary.
The primal essential b.c. come from the term $d d^star$, meaning the function needs to be zero on the boundary.
In the primal weak formulation the essential bc are imposed on the spaces directly and the
natural bc are implied the variational formulation.
For the mixed weak all bc are natural! There are no essential ones imposed on the space!

Define $hat(V) = frak(h)^(perp V) = frak(h)^perp sect V$


*$k=0$*:\
$
  -div grad u = 0
$

Hodge Decomposition is
$
  L^2 = div H0(div) perp.circle RR
$

The Poincaré inequality (called Poincaré-Neumann inequality) is
$
  norm(u)_(H^1) <= c_p norm(grad u)_(L^2)
  quad forall u in hat(H)^1
$

$hat(H^1)$ is $H^1$ without constants.


*$k=1$*:\

Hodge Decomposition is
$
  L^2 (Omega,RR^3) = grad H^1 perp.circle frak(h)^1 perp.circle curl H0(curl)
$
Violoated by existance of tunnels.


The Poincaré inequality is
$
  norm(u)_(H^curl) <= c_p norm(curl u)_(L^2)
  quad forall u in H(curl) sect cal(N)(curl)^perp
$

*$k=2$*:\

Hodge Decomposition is
$
  L^2 (Omega,RR^3) = curl H(curl) perp.circle frak(h)^2 perp.circle H0^1
$
Violated by existance of voids.

The Poincaré inequality is
$
  norm(u)_(H^div) <= c_p norm(div u)_(L^2)
  quad forall u in H(div) sect cal(N)(div)^perp
$

*$k=3$*:\


Hodge Decomposition is
$
  L^2 = div H(div)
$
Just the statement that $div$ is onto/surjective.
Never violated, because Betti numbers always zero, right?
No harmonic three forms.

The Poincaré inequality doesn't say anything.
