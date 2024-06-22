function F = analyzeTS(X,Y,win,titlename)

if nargin < 4, titlename=' '; end

mM = movmean(Y,win);
mV = movstd(Y,win);

figure('Name',titlename)
subplot(4,1,1)
plot(X,Y,'black');
hold on
plot(X,mM,'r');
xlabel('Time (weeks)');
ylabel('MOVAVG');
title('Moving average');
subplot(4,1,2)
plot(X,mV,'blue');
title('Moving deviation')
ylabel('MOVSTD')
xlabel('Time')
subplot(4,1,3)
autocorr(Y)
ylabel('Autocorrelation');
subplot(4,1,4)
parcorr(Y)
ylabel('Partial Autocorrelation');

fprintf('\nTesting for time-series stability...\n');
% Test for unit root (lag coef == 1) a.k.a. observation equivalence
% Test fails if (coef->1.0 e.g. coef=0.991)
% Augmented version removes all autocorrelation before testing
% H0: root exists
[~,pval] = adftest(Y);
fprintf('ADF:\t%.8f\n',pval);
% H0: trend-stationary (mean-reverting)
% H1: root exists (diverge from mean)
[~,pval] = kpsstest(Y);
fprintf('KPSS:\t%.8f\n',pval);
% Similar to ADF, it makes a non-parametric approach for autocorrelations
[~,pval] = pptest(Y);
fprintf('PP:\t\t%.8f\n',pval);
% H0: test if Y is a random walk
[~,pval] = vratiotest(Y);
fprintf('VRATIO:\t%.8f\n',pval);
end