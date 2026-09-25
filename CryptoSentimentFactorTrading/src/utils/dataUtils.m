function u = dataUtils()
%DATautils Utility functions for loading and merging project datasets.

u.findColumnIndex   = @findColumnIndex;
u.loadBTCData       = @loadBTCData;
u.loadSentimentData = @loadSentimentData;
u.mergeDatasets     = @mergeDatasets;
u.printDataSummary  = @printDataSummary;

end


%% ============================================================
function idx = findColumnIndex(varNames, exactCandidates, ...
                               substringCandidates, excludeCandidates)
%FINDCOLUMNINDEX Find a table column by name.

if nargin < 2
    exactCandidates = {};
end

if nargin < 3
    substringCandidates = {};
end

if nargin < 4
    excludeCandidates = {};
end

varNames = string(varNames);

cleanNames = lower(varNames);
cleanNames = erase(cleanNames, "_");
cleanNames = erase(cleanNames, " ");
cleanNames = erase(cleanNames, "-");

idx = [];


% Exact match
for k = 1:numel(exactCandidates)

    candidate = lower(string(exactCandidates{k}));

    candidate = erase(candidate, "_");
    candidate = erase(candidate, " ");
    candidate = erase(candidate, "-");

    match = find(cleanNames == candidate, 1);

    if ~isempty(match)
        idx = match;
        return;
    end

end


% Substring match
for i = 1:numel(cleanNames)

    currentName = cleanNames(i);

    excluded = false;

    for e = 1:numel(excludeCandidates)

        ex = lower(string(excludeCandidates{e}));

        ex = erase(ex, "_");
        ex = erase(ex, " ");
        ex = erase(ex, "-");

        if contains(currentName, ex)
            excluded = true;
            break;
        end

    end

    if excluded
        continue;
    end


    for j = 1:numel(substringCandidates)

        sub = lower(string(substringCandidates{j}));

        sub = erase(sub, "_");
        sub = erase(sub, " ");
        sub = erase(sub, "-");

        if contains(currentName, sub)
            idx = i;
            return;
        end

    end

end

end


%% ============================================================
function out = safeToDouble(x)
%SAFETODOUBLE Convert table values to double.

if isnumeric(x)

    out = double(x);

elseif islogical(x)

    out = double(x);

elseif iscategorical(x)

    out = str2double(string(x));

elseif isstring(x)

    out = str2double(x);

elseif ischar(x)

    out = str2double(string(x));

elseif iscell(x)

    out = str2double(string(x));

else

    try
        out = double(x);
    catch
        out = NaN(numel(x),1);
    end

end

out = out(:);

end


%% ============================================================
function dates = parseDateColumn(rawDates)
%PARSEDATECOLUMN Convert timestamps into daily datetime values.

% Already datetime
if isdatetime(rawDates)

    dates = dateshift(rawDates, 'start', 'day');
    return;

end


% Numeric timestamp
if isnumeric(rawDates)

    x = double(rawDates(:));

    dates = NaT(size(x));

    valid = isfinite(x);

    if any(valid)

        xv = x(valid);

        medValue = median(abs(xv), 'omitnan');

        if medValue > 1e11

            % Unix milliseconds
            dates(valid) = datetime( ...
                xv ./ 1000, ...
                'ConvertFrom', 'posixtime');

        elseif medValue > 1e9

            % Unix seconds
            dates(valid) = datetime( ...
                xv, ...
                'ConvertFrom', 'posixtime');

        elseif medValue > 5e5

            % MATLAB datenum
            dates(valid) = datetime( ...
                xv, ...
                'ConvertFrom', 'datenum');

        else

            dates(valid) = datetime( ...
                xv, ...
                'ConvertFrom', 'posixtime');

        end

    end

    dates = dateshift(dates, 'start', 'day');

    return;

end


% String conversion
s = string(rawDates);
s = strtrim(s);

dates = NaT(size(s));

valid = ~ismissing(s) & s ~= "";


% Remove UTC
s = erase(s, " UTC");
s = erase(s, " GMT");


% Format with microseconds
remaining = valid & isnat(dates);

if any(remaining)

    try

        dates(remaining) = datetime( ...
            s(remaining), ...
            'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSSSSS');

    catch

    end

end


% Format with milliseconds
remaining = valid & isnat(dates);

if any(remaining)

    try

        dates(remaining) = datetime( ...
            s(remaining), ...
            'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS');

    catch

    end

end


% Standard datetime parsing
remaining = valid & isnat(dates);

if any(remaining)

    try

        dates(remaining) = datetime(s(remaining));
    catch
    end

end


% Date only
remaining = valid & isnat(dates);

if any(remaining)

    try

        dates(remaining) = datetime( ...
            s(remaining), ...
            'InputFormat', 'yyyy-MM-dd');
    catch
    end

end


dates = dateshift(dates, 'start', 'day');

end


%% ============================================================
function priceTable = loadBTCData(btcFile)
%LOADBTCDATA Load Binance BTC daily OHLCV data.
%
% Expected columns:
% Open time
% Open
% High
% Low
% Close
% Volume
% Close time
% Quote asset volume
% Number of trades
% ...

fprintf('   Reading BTC CSV...\n');


if ~isfile(btcFile)

    error( ...
        'dataUtils:BTCFileNotFound', ...
        'BTC file not found:\n%s', btcFile);

end


%% ------------------------------------------------------------
% Read CSV using import options
% ------------------------------------------------------------

try

    opts = detectImportOptions( ...
        btcFile, ...
        'Delimiter', ',');

    % Preserve original column names
    try
        opts.VariableNamingRule = 'preserve';
    catch
        % Older MATLAB versions may not support this property
    end

    btc = readtable( ...
        btcFile, ...
        opts);

catch ME

    error( ...
        'dataUtils:BTCReadError', ...
        ['Unable to read BTC CSV file.\n' ...
         'Original error: %s'], ...
        ME.message);

end


if isempty(btc) || height(btc) == 0

    error( ...
        'dataUtils:BTCEmptyCSV', ...
        'BTC CSV contains no rows.');

end


fprintf('   Raw BTC rows: %d\n', height(btc));

fprintf('   Raw BTC columns: %d\n', width(btc));


%% ------------------------------------------------------------
% Display actual columns
% ------------------------------------------------------------

fprintf('   BTC columns detected:\n');

for k = 1:width(btc)

    fprintf('      %d. %s\n', ...
        k, ...
        btc.Properties.VariableNames{k});

end


%% ------------------------------------------------------------
% Find columns
% ------------------------------------------------------------

openTimeIdx = findColumnIndex( ...
    btc.Properties.VariableNames, ...
    { ...
    'Open time', ...
    'Open_time', ...
    'OpenTime'}, ...
    {'opentime'}, ...
    {});


openIdx = findColumnIndex( ...
    btc.Properties.VariableNames, ...
    {'Open'}, ...
    {'open'}, ...
    {});


highIdx = findColumnIndex( ...
    btc.Properties.VariableNames, ...
    {'High'}, ...
    {'high'}, ...
    {});


lowIdx = findColumnIndex( ...
    btc.Properties.VariableNames, ...
    {'Low'}, ...
    {'low'}, ...
    {});


closeIdx = findColumnIndex( ...
    btc.Properties.VariableNames, ...
    {'Close'}, ...
    {'close'}, ...
    {});


volumeIdx = findColumnIndex( ...
    btc.Properties.VariableNames, ...
    {'Volume'}, ...
    {'volume'}, ...
    {});


tradeIdx = findColumnIndex( ...
    btc.Properties.VariableNames, ...
    { ...
    'Number of trades', ...
    'Number_of_trades', ...
    'NumberOfTrades'}, ...
    {'numberoftrades'}, ...
    {});


%% ------------------------------------------------------------
% Fallback positions for Binance format
% ------------------------------------------------------------

if isempty(openTimeIdx)
    openTimeIdx = 1;
end

if isempty(openIdx)
    openIdx = 2;
end

if isempty(highIdx)
    highIdx = 3;
end

if isempty(lowIdx)
    lowIdx = 4;
end

if isempty(closeIdx)
    closeIdx = 5;
end

if isempty(volumeIdx)
    volumeIdx = 6;
end


%% ------------------------------------------------------------
% Display mapping
% ------------------------------------------------------------

fprintf('\n');

fprintf('   Detected column mapping:\n');

fprintf('      Date   -> %s\n', ...
    btc.Properties.VariableNames{openTimeIdx});

fprintf('      Open   -> %s\n', ...
    btc.Properties.VariableNames{openIdx});

fprintf('      High   -> %s\n', ...
    btc.Properties.VariableNames{highIdx});

fprintf('      Low    -> %s\n', ...
    btc.Properties.VariableNames{lowIdx});

fprintf('      Close  -> %s\n', ...
    btc.Properties.VariableNames{closeIdx});

fprintf('      Volume -> %s\n', ...
    btc.Properties.VariableNames{volumeIdx});


if ~isempty(tradeIdx)

    fprintf('      Trades -> %s\n', ...
        btc.Properties.VariableNames{tradeIdx});

end


%% ------------------------------------------------------------
% Parse columns
% ------------------------------------------------------------

btcDates = parseDateColumn( ...
    btc{:,openTimeIdx});


openData = safeToDouble( ...
    btc{:,openIdx});


highData = safeToDouble( ...
    btc{:,highIdx});


lowData = safeToDouble( ...
    btc{:,lowIdx});


closeData = safeToDouble( ...
    btc{:,closeIdx});


volumeData = safeToDouble( ...
    btc{:,volumeIdx});


if ~isempty(tradeIdx)

    tradeData = safeToDouble( ...
        btc{:,tradeIdx});

else

    tradeData = NaN(height(btc),1);

end


%% ------------------------------------------------------------
% Validate dates
% ------------------------------------------------------------

validDates = ~isnat(btcDates);

fprintf('\n');

fprintf('   Valid BTC dates: %d / %d\n', ...
    sum(validDates), ...
    numel(btcDates));


if sum(validDates) == 0

    error( ...
        'dataUtils:BTCDateParsing', ...
        'Unable to parse BTC dates.');

end


%% ------------------------------------------------------------
% Build standardized table
% ------------------------------------------------------------

priceTable = table( ...
    btcDates(:), ...
    openData(:), ...
    highData(:), ...
    lowData(:), ...
    closeData(:), ...
    volumeData(:), ...
    tradeData(:), ...
    'VariableNames', ...
    { ...
    'Date', ...
    'Open', ...
    'High', ...
    'Low', ...
    'Close', ...
    'Volume', ...
    'NumberOfTrades'});


%% ------------------------------------------------------------
% Remove invalid dates
% ------------------------------------------------------------

validRows = ~isnat(priceTable.Date);

priceTable = priceTable(validRows,:);


%% ------------------------------------------------------------
% Remove invalid OHLC data
% ------------------------------------------------------------

validRows = ...
    isfinite(priceTable.Open) & ...
    isfinite(priceTable.High) & ...
    isfinite(priceTable.Low) & ...
    isfinite(priceTable.Close) & ...
    priceTable.Close > 0;


priceTable = priceTable(validRows,:);


%% ------------------------------------------------------------
% Sort
% ------------------------------------------------------------

priceTable = sortrows( ...
    priceTable, ...
    'Date');


%% ------------------------------------------------------------
% Remove duplicate dates
% ------------------------------------------------------------

[~, uniqueIdx] = unique( ...
    priceTable.Date, ...
    'last');


priceTable = priceTable(uniqueIdx,:);

priceTable = sortrows( ...
    priceTable, ...
    'Date');


%% ------------------------------------------------------------
% Final validation
% ------------------------------------------------------------

if isempty(priceTable) || ...
        height(priceTable) == 0

    error( ...
        'dataUtils:BTCNoValidRows', ...
        ['No valid BTC observations remain after cleaning. ' ...
         'Please check the CSV numeric columns.']);

end


fprintf('\n');

fprintf('   Clean BTC rows: %d\n', ...
    height(priceTable));


fprintf('   BTC date range: %s to %s\n', ...
    datestr(priceTable.Date(1), 'yyyy-mm-dd'), ...
    datestr(priceTable.Date(end), 'yyyy-mm-dd'));

end


%% ============================================================
function sentTable = loadSentimentData(sentimentFile)
%LOADSENTIMENTDATA Load daily sentiment data.

fprintf('   Reading sentiment CSV...\n');


if ~isfile(sentimentFile)

    error( ...
        'dataUtils:SentimentFileNotFound', ...
        'Sentiment file not found:\n%s', ...
        sentimentFile);

end


%% Read sentiment data
try

    opts = detectImportOptions( ...
        sentimentFile, ...
        'Delimiter', ',');

    try
        opts.VariableNamingRule = 'preserve';
    catch
    end

    sent = readtable( ...
        sentimentFile, ...
        opts);

catch ME

    error( ...
        'dataUtils:SentimentReadError', ...
        ['Unable to read sentiment CSV.\n' ...
         'Original error: %s'], ...
        ME.message);

end


if isempty(sent) || ...
        height(sent) == 0

    error( ...
        'dataUtils:SentimentEmptyCSV', ...
        'Sentiment CSV contains no rows.');

end


fprintf('   Raw sentiment rows: %d\n', ...
    height(sent));


fprintf('   Sentiment columns:\n');

for k = 1:width(sent)

    fprintf('      %d. %s\n', ...
        k, ...
        sent.Properties.VariableNames{k});

end


%% ------------------------------------------------------------
% Find date
% ------------------------------------------------------------

dateIdx = findColumnIndex( ...
    sent.Properties.VariableNames, ...
    {'time','Date','date','timestamp'}, ...
    {'date','time','timestamp'}, ...
    {});


if isempty(dateIdx)
    dateIdx = 1;
end


%% ------------------------------------------------------------
% Find sentiment
% ------------------------------------------------------------

sentimentIdx = findColumnIndex( ...
    sent.Properties.VariableNames, ...
    {'sentiment','Sentiment','score'}, ...
    {'sentiment'}, ...
    {});


if isempty(sentimentIdx)
    sentimentIdx = 2;
end


%% ------------------------------------------------------------
% Find sample size
% ------------------------------------------------------------

sampleIdx = findColumnIndex( ...
    sent.Properties.VariableNames, ...
    { ...
    'sample_size', ...
    'SampleSize', ...
    'sample size', ...
    'sample', ...
    'count'}, ...
    {'samplesize'}, ...
    {});


%% ------------------------------------------------------------
% Parse
% ------------------------------------------------------------

sentDates = parseDateColumn( ...
    sent{:,dateIdx});


sentimentValues = safeToDouble( ...
    sent{:,sentimentIdx});


if ~isempty(sampleIdx)

    sampleValues = safeToDouble( ...
        sent{:,sampleIdx});

else

    sampleValues = NaN(height(sent),1);

end


%% ------------------------------------------------------------
% Create table
% ------------------------------------------------------------

sentTable = table( ...
    sentDates(:), ...
    sentimentValues(:), ...
    sampleValues(:), ...
    'VariableNames', ...
    {'Date','Sentiment','SampleSize'});


%% ------------------------------------------------------------
% Clean
% ------------------------------------------------------------

validRows = ...
    ~isnat(sentTable.Date) & ...
    isfinite(sentTable.Sentiment);


sentTable = sentTable(validRows,:);


sentTable = sortrows( ...
    sentTable, ...
    'Date');


%% Remove duplicates
[~, uniqueIdx] = unique( ...
    sentTable.Date, ...
    'last');


sentTable = sentTable(uniqueIdx,:);


sentTable = sortrows( ...
    sentTable, ...
    'Date');


%% Final validation
if isempty(sentTable) || ...
        height(sentTable) == 0

    error( ...
        'dataUtils:SentimentNoValidRows', ...
        'No valid sentiment observations remain after cleaning.');

end


fprintf('   Clean sentiment rows: %d\n', ...
    height(sentTable));


fprintf('   Sentiment date range: %s to %s\n', ...
    datestr(sentTable.Date(1), 'yyyy-mm-dd'), ...
    datestr(sentTable.Date(end), 'yyyy-mm-dd'));

end


%% ============================================================
function merged = mergeDatasets(priceTable, sentTable)
%MERGEDATASETS Inner join BTC and sentiment data by Date.

if isempty(priceTable)

    error( ...
        'dataUtils:EmptyPriceTable', ...
        'BTC price table is empty.');

end


if isempty(sentTable)

    error( ...
        'dataUtils:EmptySentimentTable', ...
        'Sentiment table is empty.');

end


priceTable = sortrows( ...
    priceTable, ...
    'Date');


sentTable = sortrows( ...
    sentTable, ...
    'Date');


merged = innerjoin( ...
    priceTable, ...
    sentTable, ...
    'Keys', ...
    'Date');


merged = sortrows( ...
    merged, ...
    'Date');


if isempty(merged) || ...
        height(merged) == 0

    warning( ...
        'dataUtils:NoDateOverlap', ...
        'No overlapping dates found.');

else

    fprintf('\n');

    fprintf('   Overlapping dates: %d\n', ...
        height(merged));


    fprintf('   Merged date range: %s to %s\n', ...
        datestr(merged.Date(1), 'yyyy-mm-dd'), ...
        datestr(merged.Date(end), 'yyyy-mm-dd'));

end

end


%% ============================================================
function printDataSummary(tbl, titleText)
%PRINTDATASUMMARY Print dataset summary.

fprintf('\n');

fprintf('------------------------------------------------------------\n');

fprintf(' %s\n', titleText);

fprintf('------------------------------------------------------------\n');


if isempty(tbl)

    fprintf(' Table is empty.\n');

    fprintf('------------------------------------------------------------\n\n');

    return;

end


fprintf(' Rows    : %d\n', ...
    height(tbl));


fprintf(' Columns : %d\n', ...
    width(tbl));


fprintf('\nVariables:\n');

for k = 1:width(tbl)

    fprintf('  - %s\n', ...
        tbl.Properties.VariableNames{k});

end


if ismember( ...
        'Date', ...
        tbl.Properties.VariableNames)

    validDates = ~isnat(tbl.Date);

    if any(validDates)

        d = tbl.Date(validDates);

        fprintf('\nDate range:\n');

        fprintf('  Start : %s\n', ...
            datestr(min(d), 'yyyy-mm-dd'));

        fprintf('  End   : %s\n', ...
            datestr(max(d), 'yyyy-mm-dd'));

    end

end


fprintf('------------------------------------------------------------\n\n');

end