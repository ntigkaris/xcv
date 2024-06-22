function [rateConfInt,meandata] = RunMeanBootstrap(dailydata,EUrate)

    M = 1000;
    N = length(dailydata);
    alpha = 0.05;
    dailydata = sort(dailydata);
    meandata = mean(dailydata);

    % bootstrap
    for i=1:M
        
        temp = dailydata(unidrnd(N,N,1));
        weekrate(i) = mean(temp);

    end

    indxLow = round(M*alpha/2); % find confidence limits
    indxUp = round(M*(1-alpha/2));
    weekrate = weekrate';
    weekrate = sort(weekrate);
    rateConfInt = [weekrate(indxLow),weekrate(indxUp)];
    % significant difference
    fprintf('Significance difference: [ %+.4f \t %+.4f ]\n',...
        EUrate - rateConfInt(1),EUrate - rateConfInt(2));
    
end