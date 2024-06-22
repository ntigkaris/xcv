%% Coronavirus Data Analysis
% Ntigkaris E. Alexandros
% MATLAB R2018a

% In this project we will perform data analysis on coronavirus european and
% greek data from January 2020 to December 2021. Data are taken from the
% official ECDC site [ https://www.ecdc.europa.eu/en/covid-19/data ], the
% official EODY site [ https://eody.gov.gr/ ] and the research page [
% https://www.stelios67pi.eu/ ].

% Project is divided in four tasks.

% Task 01: We analyze GR positivity rate and the probability distribution
% of 25 EU countries, during Greece's maximum rate in the end of 2020 and
% 2021. Here we are interested in time periods W45-W50 2020 and W45-W50
% 2021, as the end of both years marked significant change in the course of
% the pandemic.

% Task 02: We perform a Kolmogorov-Smirnov test, to see whether both 2020 &
% 2021 data come from the same distribution. We run both parametric and non
% -parametric tests to obtain confidence intervals.

% Task 03: We run non-parametric bootstrap, testing how statistically
% significant GR positivity rate is, compared to the EU one. We pick a time
% period of 12 weeks for our test, starting from September 2020 and ending 
% around the end of November 2020. Again, this period was extremely crucial
% to the pandemic course for Greece.

% Task 04: We analyze Greece positivity rate full trajectory as a time
% series. We test the series for stability using various algorithms and
% graphical methods. Stability in a time series is crucial if we want to go
% forward and make predictions.

%% -- start fresh --

clear all; close all; clc

% -- load our data --

load EuropeanCountries.mat
load ECDC-7Days-Testing.mat
load FullEodyData.mat

%% *** TASK 01 ***

% In this segment, we analyze the positivity rate for Greece during the
% 45th to 50th week of 2020 and 2021 respectively. From those dates we then
% extract the week where Greece had maximum positivity rate. At that time
% we also get the positivity rates for 25 other european countries (see
% EuropeanCountries.mat) and we infer the distribution that coronavirus
% EU positivity rate follows in general during that time.

mycountry = 'Greece';
indx = find(country==mycountry & level=='national'); % get chosen country's indeces from Testing file
unique(country); % check if Testing file contains more countries than we want to
grRate = positivity_rate(indx); % get Greece total positivity rate

% -- find 2020 W45-W50 positivity rates --
indx_weeks_Y20 = find(year_week(indx)>'2020-W44' & year_week(indx)<'2020-W51');
int_Y20 = indx(indx_weeks_Y20); % find 2020W45-W50 indeces for my country
max_pos_Y20 = max(positivity_rate(int_Y20));
maxposWeekY20 = year_week(find(positivity_rate == max_pos_Y20)); % find week where positivity was max

% get all 25 countries' indeces of selected date on a national level
indx_Y20 = find(year_week == maxposWeekY20 & level=='national' & ismember(country,Country));
length(indx_Y20); % we see here that we got data for 23 out of 25 countries
setxor(country(indx_Y20),Country); % apparently France and Latvia are
% missing therefore we will search for the closest data to that date
indxFR = find(country=='France' & level=='national' & year_week>='2020-W43' & year_week<=maxposWeekY20);
indxLV = find(country=='Latvia' & level=='national' & year_week>='2020-W43' & year_week<=maxposWeekY20);
data_maxPos_W45W50Y20 = positivity_rate(indx_Y20); % get positivity rate data for that date
data_maxPos_W45W50Y20(end+1) = positivity_rate(indxFR); % append the missing data
data_maxPos_W45W50Y20(end+1) = positivity_rate(indxLV);

% -- find 2021 W45-W50 positivity rates --
indx_weeks_Y21 = find(year_week(indx)>'2021-W44' & year_week(indx)<'2021-W51');
int_Y21 = indx(indx_weeks_Y21); % find 2021W45-W50 indeces for my country
max_pos_Y21 = max(positivity_rate(int_Y21));
maxposWeekY21 = year_week(find(positivity_rate == max_pos_Y21)); % find week where positivity was max

% get all 25 countries' indeces of selected date on a national level
indx_Y21 = find(year_week == maxposWeekY21 & level=='national' & ismember(country,Country));
length(indx_Y21); % in 2021 no country's data are missing
data_maxPos_W45W50Y21 = positivity_rate(indx_Y21); % get positivity rate data for that date

% -- positivity rate distributions around chosen dates --

bins = 5; % given the small sample of our data, a range of (5,20) is preferred

figure
plot(1:length(indx), positivity_rate(indx),'black');
x1 = find(indx == find(positivity_rate == max_pos_Y20));
x2 = find(indx == find(positivity_rate == max_pos_Y21));
hold on
plot([x1,x1],[0,max_pos_Y20],'b--');
plot([x2,x2],[0,max_pos_Y21],'b--');
plot([x1,x1],[max_pos_Y20,max_pos_Y20],'ro');
plot([x2,x2],[max_pos_Y21,max_pos_Y21],'ro');
ylabel('Positivity rate (%)');
xlabel('Time period (weeks)');
legend('Positivity rate','max rate W45-W50 Y20','max rate W45-W50 Y21');
title('Greece positivity rate, years 2020 & 2021');

figure
subplot(1,2,1)
histogram(data_maxPos_W45W50Y20,bins,'FaceColor',[0 0.9 0.3],'EdgeAlpha',0.7)
title(sprintf('%s Positivity Rate',maxposWeekY20));
ylabel('Frequencies');
xlabel('Values');
subplot(1,2,2)
histogram(data_maxPos_W45W50Y21,bins,'FaceColor',[0 0.9 0.3],'EdgeAlpha',0.7)
title(sprintf('%s Positivity Rate',maxposWeekY21));
ylabel('Frequencies');
xlabel('Values');

% from the histograms we can infer that the data do not come from a uniform
% distribution since the values do not have the same frequency of
% appearance. Therefore in the next segment, we will investigate if data
% are fit well into a normal or an exponential distribution.

% -- fit distributions --

h20N = fitdist(data_maxPos_W45W50Y20,'Normal'); % normal fit statistics / 2020
fprintf('Normal fit :\t\t pos. rate 2020 :\t mu: %.4f \t sigma: %.4f\n',...
    mean(h20N),std(h20N));
h21N = fitdist(data_maxPos_W45W50Y21,'Normal'); % normal fit statistics / 2021
fprintf('Normal fit :\t\t pos. rate 2021 :\t mu: %.4f \t sigma: %.4f\n',...
    mean(h21N),std(h21N));
h20E = fitdist(data_maxPos_W45W50Y20,'Exponential'); % exp fit statistics / 2020
fprintf('Exponential fit :\t pos. rate 2020 :\t mu: %.4f \t sigma: %.4f\n',...
    mean(h20E),std(h20E));
h21E = fitdist(data_maxPos_W45W50Y21,'Exponential'); % exp fit statistics / 2021
fprintf('Exponential fit :\t pos. rate 2021 :\t mu: %.4f \t sigma: %.4f\n',...
    mean(h21E),std(h21E));

figure
subplot(2,2,1)
h1 = histfit(data_maxPos_W45W50Y20,bins,'normal');
title(sprintf('%s Normal fit on Positivity Rate',maxposWeekY20));
ylabel('Frequencies');
xlabel('Values');

subplot(2,2,2)
h2 = histfit(data_maxPos_W45W50Y20,bins,'exponential');
title(sprintf('%s Exponential fit on Positivity Rate',maxposWeekY20));
ylabel('Frequencies');
xlabel('Values');

subplot(2,2,3)
h3 = histfit(data_maxPos_W45W50Y21,bins,'normal');
title(sprintf('%s Normal fit on Positivity Rate',maxposWeekY21));
ylabel('Frequencies');
xlabel('Values');
subplot(2,2,4)
h4 = histfit(data_maxPos_W45W50Y21,bins,'exponential');
title(sprintf('%s Exponential fit on Positivity Rate',maxposWeekY21));
ylabel('Frequencies');
xlabel('Values');

set([h1(1) h2(1) h3(1) h4(1)],'facecolor',[0 0.9 0.3]);
set([h1(2) h2(2) h3(2) h4(2)],'color',[0.15 0.15 0.15]);
set([h1(2) h2(2) h3(2) h4(2)],'LineWidth',1);

%% Conclusions:
% We can clearly see from the histograms that our data do not follow a
% UNIFORM distribution. We should also note that given the small sample
% (25), our data could potentially approximate a NORMAL distribution in
% regards with CLT, if our sample was larger. Taking into account now the
% fit of the parametric distributions, we see that both 2020 and 2021 data
% are described sufficiently by the EXPO distribution. That is well
% expected, given the fact that coronavirus cases tend to evolve
% exponentially and given that the timeline of our readings is around
% December (W45-W50), a month which was crucial both in 2020 and 2021, for
% the evolution of the pandemic course.

%% *** TASK 02 ***

% We assume a null hypothesis that 2020 data & 2021 data come from the same
% distribution. To test that hypothesis, we use the Kolmogorov-Smirnov
% test, under which the maximum difference between the empirical
% distribution functions of data is measured. We calculate the observed 
% statistic in our sample. We run a permutation test to assess the confidence
% interval for the statistic. If observed statistic is within the interval 
% for significance level alpha, we retain null hypothesis. Otherwise we reject it.

% -- Kolmogorov-Smirnov Test --

joint = sort(cat(1,data_maxPos_W45W50Y20,data_maxPos_W45W50Y21)); % sort and concat '20 and '21 data
M = 1000; % M randomized samples
alpha = 0.05;
N = length(joint);

%  -- Parametric: obtain observed statistic --

data_maxPos_W45W50Y20 = sort(data_maxPos_W45W50Y20);
data_maxPos_W45W50Y21 = sort(data_maxPos_W45W50Y21);
sumval2020 = NaN*zeros(N,1);
sumval2021 = NaN*zeros(N,1);
for i=1:N
    % calculate how many times an observation appears in sample
    sumval2020(i,1) = sum(data_maxPos_W45W50Y20(:)==joint(i));
    sumval2021(i,1) = sum(data_maxPos_W45W50Y21(:)==joint(i));
end
edf2020 = cumsum(sumval2020)./(N/2); % calculate the EDFs
edf2021 = cumsum(sumval2021)./(N/2);
tobs = max(abs(edf2020 - edf2021)); % calculate max distance between the 2 EDFs

% -- plot both empirical functions

figure
plot(joint,edf2020);
hold on
plot(joint,edf2021);
title('Empirical distribution functions (Kolmogorov-Smirnov Test)');
ylim([0,1.05]);
ylabel('Probability');
xlabel('observed data');
indx = find(abs(edf2020 - edf2021)==tobs); % KS statistic
y1 = edf2020(indx(1));
y2 = edf2021(indx(1));
x0 = joint(indx(1));
plot([x0,x0],[y1,y2],'black--'); % plot KS statistic
legend('EDF - 2020','EDF - 2021','KS statistic');

% -- Non-parametric: Permutation test --

t = NaN*zeros(M,1); % permutation statistics array
for i=1:M

    temp = joint(randperm(N)); % randomized joint vector
    X = sort(temp(1:N/2)); % size N/2
    Y = sort(temp(N/2+1:end)); % size N/2
    sumvalX = NaN*zeros(N,1);
    sumvalY = NaN*zeros(N,1);
    for j=1:N
        % calculate how many times an observation appears in sample
        sumvalX(j,1) = sum(X(:)==joint(j));
        sumvalY(j,1) = sum(Y(:)==joint(j));
    end
    
    edfX = cumsum(sumvalX)./(N/2); % calculate the EDFs
    edfY = cumsum(sumvalY)./(N/2);
    t(i,1) = max(abs(edfX - edfY)); % calculate max distance between the 2 EDFs
    
end

t = sort(t); % sort array
tL = t(round(alpha/2 *M)); % get interval limits for significance alpha
tU = t(round((1-alpha/2) *M));
% if observed statistic not within the interval derived by the permutation
% test, reject null hypothesis. Otherwise, do not reject it.
fprintf('\nKolmogorov-Smirnov Test results:\n');
if tobs<tL || tobs>tU, fprintf('stat: %.2f not within interval [%.2f,%.2f], reject null hypothesis\n',...
        tobs,tL,tU);
else, fprintf('stat: %.2f is within interval [%.2f,%.2f], retain null hypothesis\n'...
        ,tobs,tL,tU);
end

%% Conclusions:
% Null hypothesis is retained. Both data come from the same distribution.

%% *** TASK 03 ***

% In this segment we will test how statistically significant greek
% positivity rate is, compared with the European one. To test this we will
% pick a time period of 12 weeks, starting from 2020/08/31 and ending to
% 2020/11/22. We pick these dates, as this period was marked in Greece by a
% significant rise in coronavirus cases.

starting_week = '2020-W36'; % 2020-08-31 to 2020-09-06
ending_week	= '2020-W47'; %	2020-11-16 to 2020-11-22
startDate = find(Date=='2020-08-31'); % find correspoding date in EODY file
endDate = find(Date=='2020-11-22');

% -- fixing nan values --

nanIndex = find(isnan(Rapid_Tests(startDate:endDate)));
% we can see we have some missing RAPID TEST data for Greece
% especially for the time interval we are interested in

% thankfully in ECDC-7Days-Testing.mat we are given info about the total
% tests on Greece per week. By taking these values, dividing them by 7
% (daily rate) and substracting the daily PCR tests, we'll get an estimate
% for daily rapid testing in greece at the missing dates
startIndx = find(country=='Greece' & level == 'national' & year_week==starting_week);
endIndx = find(country=='Greece' & level == 'national' & year_week==ending_week);
totalweektests = tests_done(startIndx:endIndx); % totals tests in 12 weeks

% calculate daily tests' estimate in 12 weeks period
j=1;
for i=1:nanIndex(end)
    totaldailytests(i) = totalweektests(j)/7.0; % daily tests estimate
    if mod(i,7)==0 % change week
        j = j+1;
    end
end
totaldailytests = totaldailytests';

% now since the testing data in EODY file are given cumulatively, we will
% normalize them by subtracting the previous value, so we know exactly how
% many tests were perfomed every single day.
NormPCR(1) = PCR_Tests(1);
NormRapid(1) = Rapid_Tests(1);
for i=2:length(NewCases)
    NormPCR(i) = PCR_Tests(i) - PCR_Tests(i-1);
    NormRapid(i) = Rapid_Tests(i) - Rapid_Tests(i-1);
end
NormPCR = NormPCR'; NormRapid = NormRapid';

% now that our testing data are normalized, we will take the estimated
% daily tests data and subtract the daily PCR data to get an estimate for
% the missing Rapid tests values.

% Small note:
% we are taking the absolute value of the subtraction since total data are
% estimates so sometimes they can oscillate +- 3.000 from the real measured
% number and thus give a negative value result when we underestimate it.
j = 1;
for i=startDate:startDate+nanIndex(end)
    NormRapid(i)=abs(totaldailytests(j) - NormPCR(i));
    j = j+1;
    if j>nanIndex(end)
        break
    end
end
NormRapid(startDate+nanIndex(end)) = (NormRapid(startDate+nanIndex(end)-1)...
    + NormRapid(startDate+nanIndex(end)+1)) /2;
sum(isnan(NormRapid(startDate:endDate))); % nan values are gone

% -- positivity rates for Greece & EU --

% now that we have fixed the nan values in our data, we can calculate the
% daily positivity rate for Greece
grposrate = NewCases(startDate:endDate)*100./(NormPCR(startDate:endDate)+...
    NormRapid(startDate:endDate));

% after we've defined Greece's 12-week data, we're gonna load EU's 12-week
% data taken manually from the site's graph: https://www.stelios67pi.eu/testing.html

load EUpositivityRate.mat

% -- Bootstrap test --

confInterval = zeros(2,length(euposrate)); % confidence intervals
grweekrate = zeros(length(euposrate),1); % weekly greek positivity rate

startRange = 1; endRange = 7;

fprintf('\nBootstrap (zero or negative difference implies insignificant result):\n');
for i=1:length(euposrate)
    [confInterval(:,i),grweekrate(i,1)] = ...
        RunMeanBootstrap(grposrate(startRange:endRange),euposrate(i));
    startRange = endRange + 1; % change week
    endRange = startRange + 6;
end

figure
xPlot = 1:12;
plot(xPlot,grweekrate,'black-');
hold on
plot(xPlot,euposrate,'r-');
plot(xPlot,confInterval(1,:),'g--');
plot (xPlot,confInterval(2,:),'g--');
title('Weekly positivity rates for GR - EU');
legend('GR rate','EU rate','confidence interval');
ylabel('Weekly positivity rate (%)');
xlabel('Time period (Weeks)');

%% Conclusion:
% We observe that European weekly rate remains statistically significant
% concerning Greek weekly positivity rate throughout all 12-week duration.
% However in the end of the graph where the confidence interval grows, the
% EU rate may become insignificant at later dates.

%% *** TASK 04 ***

% In this segment we analyze the full trajectory of Greece's positivity
% rate, testing the series for stability using various algorithms and
% plotting the moving average, moving standard deviation and
% autocorrelation oscillations.

analyzeTS(1:length(grRate),grRate,5,...
    'Greece positivity rate time series analysis');
