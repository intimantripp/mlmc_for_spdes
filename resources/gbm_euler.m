%
% This tests the use of multilevel Monte Carlo for European call 
% and digital call options, based on an Euler-Maruyama numerical
% approximation of the Geometric Brownian Motion SDE
%
% This produces figures which are presented in section 5 in the 
% Acta Numerica article
%

function gbm

close all; clear all;

%
% use global variables to pass parameters into gbm_l function
%

global S0 K T r sig option

S0  = 100;   % initial asset value
K   = 100;   % strike
T   = 1;     % maturity
r   = 0.05;  % risk-free interest rate
sig = 0.2;   % volatility

%
% call mlmc_test to perform MLMC tests
%

nvert = 3;      % plotting option (1 for slides, 2 for papers, 3 for full set)
M     = 2;      % refinement cost factor (2^gamma in general MLMC Thm)

N     = 100000; % samples for convergence tests
L     = 8;      % levels for convergence tests 

N0    = 1000;   % initial number of samples on first 3 MLMC levels
Eps   = [ 0.005 0.01 0.02 0.05 0.1 ];  % desired accuracies for MLMC calcs
%Eps   = [ 0.1 ];  % desired accuracies for MLMC calcs

%------- European call option ------------

option = 1;
figs = mlmc_test(@gbm_l, M, N,L, N0,Eps, nvert);

subplot(3,2,5)
axis([0 8 1e3 1e8])
legend('0.005','0.01','0.02','0.05','0.1')

subplot(3,2,6)
axis([0.005 0.1 500 5e4])

print('-deps2','euler_call.eps')

%------- Digital call option ------------

option = 2;     % European digital call option
figs = mlmc_test(@gbm_l, M, N,L, N0,Eps, nvert);

subplot(3,2,5)
axis([0 8 1e3 1e8])
legend('0.005','0.01','0.02','0.05','0.1')

subplot(3,2,6)
axis([0.005 0.1 500 5e4])

print('-deps2','euler_digital.eps')

%-------------------------------------------------------
%
% level l estimator for GBM with factor 2 refinement
%

function [sum1 sum2] = gbm_l(l,N)

global S0 K T r sig option

M  = 2;

nf = M^l;
nc = nf/M;

hf = T/nf;
hc = T/nc;

sum1(1:4) = 0;
sum2(1:2) = 0;

for N1 = 1:10000:N
  N2 = min(10000,N-N1+1);

  Sf = S0*ones(1,N2);
  Sc = Sf;
  Pc = zeros(1,N2);

  if l==0
    dWf = sqrt(hf)*randn(1,N2);
    Sf  = Sf + r*Sf*hf + sig*Sf.*dWf;

  else
    for n = 1:nc
      dWc = zeros(1,N2);
      for m = 1:M
        dWf = sqrt(hf)*randn(1,N2);
        dWc = dWc + dWf;
        Sf  = Sf + r*Sf*hf + sig*Sf.*dWf;
      end
      Sc = Sc + r*Sc*hc + sig*Sc.*dWc;
    end
  end

  if (option==1)
    Pf = exp(-r*T)*max(0,Sf-K);
    if (l>0)
      Pc = exp(-r*T)*max(0,Sc-K);
    end
  elseif (option==2)
    Pf = exp(-r*T)*10*0.5*(1+sign(Sf-K));
    if (l>0)
      Pc = exp(-r*T)*10*0.5*(1+sign(Sc-K));
    end
  end

  sum1(1) = sum1(1) + sum(Pf-Pc);
  sum1(2) = sum1(2) + sum((Pf-Pc).^2);
  sum1(3) = sum1(3) + sum((Pf-Pc).^3);
  sum1(4) = sum1(4) + sum((Pf-Pc).^4);
  sum2(1) = sum2(1) + sum(Pf);
  sum2(2) = sum2(2) + sum(Pf.^2);
end
