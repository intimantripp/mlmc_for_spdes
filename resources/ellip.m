%
% This tests the use of multilevel Monte Carlo for a simple
% 1D elliptic PDE with random coefficients and random forcing:
%
% u(0)=0, u(1)=0, (c u')'(x)= - 50 Z^2
% where Z = N(0,1)                    (Normal with unit variance)
% and c(x) = 1 + a*x with a = U(0,1)  (uniformly distributed on (0,1))
% 
% the output is P = \int_0^1 u dx
%

function ellip

close all; clear all;

%
% call mlmc_test to perform MLMC tests
%

nvert = 3;      % plotting option (1 for slides, 2 for papers, 3 for full set)
M     = 2;      % refinement cost factor (2^gamma in general MLMC Thm)

N     = 10000;  % samples for convergence tests
L     = 8;      % levels for convergence tests 

N0    = 200;    % initial number of samples on first 3 MLMC levels
Eps   = [ 0.005 0.01 0.02 0.05 0.1 ];  % desired accuracies for MLMC calcs

figs = mlmc_test(@ellip_l, M, N,L, N0,Eps, nvert);

subplot(3,2,1)
axis([0 8 -30 10])

subplot(3,2,5)
axis([0 6 100 1e8])
legend('0.005','0.01','0.02','0.05','0.1')

subplot(3,2,6)
axis([0.005 0.1 50 1e4])

print('-deps2','ellip.eps')


%-------------------------------------------------------
%
% level l estimator for elliptic solver
%

function [sum1 sum2] = ellip_l(l,N)

% global

nf  = 2^(l+1);
hf  = 1/nf;
cf  = ones(nf,1);
A0f = hf^(-2)*spdiags([cf(2:end) -cf(2:end)-cf(1:end-1) cf(1:end-1)],-1:1,nf-1,nf-1);
cf  = ((1:nf)'-0.5)*hf;
A1f = hf^(-2)*spdiags([cf(2:end) -cf(2:end)-cf(1:end-1) cf(1:end-1)],-1:1,nf-1,nf-1);
cf  = ones(nf-1,1);

if l>0
  nc = nf/2;
  hc = 1/nc;
  cc  = ones(nc,1);
  A0c = hc^(-2)*spdiags([cc(2:end) -cc(2:end)-cc(1:end-1) cc(1:end-1)],-1:1,nc-1,nc-1);
  cc  = ((1:nc)'-0.5)*hc;
  A1c = hc^(-2)*spdiags([cc(2:end) -cc(2:end)-cc(1:end-1) cc(1:end-1)],-1:1,nc-1,nc-1);
  cc  = ones(nc-1,1);
end

sum1(1:4) = 0;
sum2(1:2) = 0;

for N1 = 1:N         % compute samples 1 at a time
  U  = rand(1,1);
  Z  = randn(1,1);

  uf = - (A0f+U*A1f) \ (50*Z^2*cf);
  Pf = sum(hf*uf);

  if l==0
    Pc = 0;
  else
    uc = - (A0c+U*A1c) \ (50*Z^2*cc);
    Pc = sum(hc*uc);
  end

  sum1(1) = sum1(1) + Pf-Pc;
  sum1(2) = sum1(2) + (Pf-Pc)^2;
  sum1(3) = sum1(3) + (Pf-Pc)^3;
  sum1(4) = sum1(4) + (Pf-Pc)^4;
  sum2(1) = sum2(1) + Pf;
  sum2(2) = sum2(2) + Pf.^2;
end
