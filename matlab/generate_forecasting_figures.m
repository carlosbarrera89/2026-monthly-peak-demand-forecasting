function outputFolder = generate_forecasting_figures(resultsZip, outputFolder)
%GENERATE_FORECASTING_FIGURES Export manuscript Figures 3--13.
% Run this file and select RESULTS_TO_SHARE.zip, or pass an extracted run:
%   generate_forecasting_figures('/path/to/RESULTS_TO_SHARE.zip')
%   generate_forecasting_figures('/path/to/run', '/path/to/figure_output')
%
% Exports PNG at 600 dpi and vector PDF with the Overleaf filenames.
% Requires MATLAB R2020b or later and Times New Roman. See ../README.txt.

    if nargin < 1 || isempty(resultsZip)
        try
            [fileName, folderName] = uigetfile('*.zip', ...
                'Select RESULTS_TO_SHARE.zip or a compatible results ZIP');
        catch
            fileName = input('Enter the full results ZIP or extracted-folder path: ', 's');
            folderName = '';
            if isempty(fileName)
                fileName = 0;
            end
        end
        if isequal(fileName, 0)
            outputFolder = '';
            fprintf('No ZIP selected. Nothing was changed.\n');
            return;
        end
        resultsZip = fullfile(folderName, fileName);
    end
    resultsZip = char(resultsZip);
    % Resolve relative paths before a fallback command changes directory.
    if ispc
        absoluteInput = ~isempty(regexp(resultsZip,'^[A-Za-z]:[\\/]|^[\\/]{2}','once'));
    else
        absoluteInput = startsWith(resultsZip, '/');
    end
    if ~absoluteInput
        resultsZip = fullfile(pwd, resultsZip);
    end
    inputIsFolder = isfolder(resultsZip);
    if ~inputIsFolder && ~isfile(resultsZip)
        error('ForecastFigures:MissingZIP', 'Results ZIP or folder not found: %s', resultsZip);
    end
    [zipFolder, zipStem, zipExtension] = fileparts(resultsZip);
    if ~inputIsFolder && ~strcmpi(zipExtension, '.zip')
        error('ForecastFigures:ZIPRequired', 'Select a results ZIP or its extracted folder, not a PDF or image.');
    end
    if verLessThan('matlab', '9.9')
        error('ForecastFigures:MATLABVersion', ...
            'This script requires MATLAB R2020b or later.');
    end

    cfg = figureSettings();
    installedFonts = listfonts;
    if ~any(strcmpi(installedFonts, cfg.fontName))
        error('ForecastFigures:FontMissing', ...
            ['Times New Roman is not available to MATLAB. Install or enable ', ...
             'that font, restart MATLAB, and run again. No substitute font ', ...
             'has been selected.']);
    end

    if nargin < 2 || isempty(outputFolder)
        stamp = datestr(now, 'yyyymmdd_HHMMSS');
        outputFolder = fullfile(zipFolder, ['MATLAB_Figures_' zipStem '_' stamp]);
    end
    outputFolder = char(outputFolder);
    if ispc
        absoluteOutput = ~isempty(regexp(outputFolder,'^[A-Za-z]:[\\/]|^[\\/]{2}','once'));
    else
        absoluteOutput = startsWith(outputFolder, '/');
    end
    if ~absoluteOutput
        outputFolder = fullfile(pwd, outputFolder);
    end
    if isfolder(outputFolder)
        existing = dir(outputFolder);
        existing = existing(~ismember({existing.name}, {'.', '..'}));
        if ~isempty(existing)
            error('ForecastFigures:OutputNotEmpty', ...
                'Choose an empty output folder to avoid replacing files: %s', outputFolder);
        end
    else
        mkdir(outputFolder);
    end
    figureFolder = fullfile(outputFolder, 'Figures');
    dataFolder = fullfile(outputFolder, 'PlotData');
    mkdir(figureFolder);
    mkdir(dataFolder);

    status = struct('status', 'RUNNING', ...
        'results_zip', resultsZip, 'output_folder', outputFolder, ...
        'matlab_release', version('-release'), 'matlab_version', version, ...
        'font', cfg.fontName, 'png_dpi', cfg.dpi, ...
        'first_manuscript_figure', 3, 'last_manuscript_figure', 13, ...
        'forecast_values_modified', false, 'exporter_version', '1.2.0', ...
        'requires_jvm', false, 'started_at', datestr(now, 31));
    writeJSON(fullfile(outputFolder, 'export_status.json'), status);
    workFolder = tempname;
    mkdir(workFolder);
    workCleanup = onCleanup(@() removeTemporaryFolder(workFolder)); %#ok<NASGU>
    ownedFigures = gobjects(0);
    sourceRecords = struct('role', {}, 'path', {}, 'rows', {});
    try
        fprintf('\nReading %s\n', resultsZip);
        if inputIsFolder
            inputRoot = resultsZip;
            status.extraction_method = 'Already extracted results folder';
        else
            status.extraction_method = extractZIP(resultsZip, workFolder);
            inputRoot = workFolder;
        end
        runFolder = locateRun(inputRoot);
        [~,status.source_run] = fileparts(runFolder);
        fprintf('Results folder: %s\n', status.source_run);
        verifyRunStatus(runFolder);

        countries = {'spain', 'france'};
        displayNames = {'Spain', 'France'};
        D = cell(1, 2);
        for k = 1:2
            [D{k}, records] = readCountryData(runFolder, countries{k});
            sourceRecords = [sourceRecords, records]; %#ok<AGROW>
            writePlotData(D{k}, dataFolder, countries{k});
        end
        if ~isequaln(D{1}.calendar, D{2}.calendar)
            error('ForecastFigures:CalendarsDiffer', ...
                'Spain and France have different evaluation calendars. Do not merge them into Figure 3.');
        end

        figureMap = outputNames();
        figures = cell(11, 1);
        figures{1} = plotCalendar(D{1}.calendar, cfg);
        ownedFigures(end + 1) = figures{1};
        for k = 1:2
            figures{1 + k} = plotBacktest(D{k}, displayNames{k}, cfg);
            ownedFigures(end + 1) = figures{1 + k};
            figures{3 + k} = plotMonthly(D{k}, displayNames{k}, cfg);
            ownedFigures(end + 1) = figures{3 + k};
            figures{5 + k} = plotAnnual(D{k}, displayNames{k}, cfg);
            ownedFigures(end + 1) = figures{5 + k};
            figures{7 + k} = plotContributions(D{k}, displayNames{k}, cfg);
            ownedFigures(end + 1) = figures{7 + k};
            figures{9 + k} = plotSensitivity(D{k}, displayNames{k}, cfg);
            ownedFigures(end + 1) = figures{9 + k};
        end

        for k = 1:height(figureMap)
            baseName = char(figureMap.file_stem(k));
            fprintf('Exporting manuscript Figure %d: %s\n', ...
                figureMap.manuscript_figure(k), baseName);
            exportOneFigure(figures{k}, figureFolder, baseName, cfg);
            if ~cfg.keepFiguresOpen
                close(figures{k});
            end
        end
        writetable(figureMap, fullfile(outputFolder, 'figure_filename_map.csv'));
        for k = 1:numel(sourceRecords)
            sourceRecords(k).path = strrep(sourceRecords(k).path, [inputRoot filesep], '');
        end
        writetable(struct2table(sourceRecords), ...
            fullfile(outputFolder, 'input_sources.csv'));
        [status.bundle_method,bundleMessage] = createFigureZIP(outputFolder);
        status.bundle_created = ~strcmp(status.bundle_method,'not_created');

        status.status = 'COMPLETED';
        if ~status.bundle_created
            status.status = 'COMPLETED_WITH_WARNINGS';
            status.packaging_warning = bundleMessage;
            warning('ForecastFigures:OutputZIP', '%s', bundleMessage);
        end
        status.completed_at = datestr(now, 31);
        status.number_of_png_files = 11;
        status.number_of_pdf_files = 11;
        status.input_checks_passed = true;
        status.history_source = 'Observed input files, checked against benchmark targets';
        status.export_method = 'exportgraphics: PNG at 600 dpi; PDF ContentType=vector';
        writeJSON(fullfile(outputFolder, 'export_status.json'), status);
        fprintf('\nFinished. Figures 3--13 were exported.\n%s\n', figureFolder);
        fprintf('Upload the contents of Figures/ to the Figures/ folder in Overleaf.\n');
        fprintf('Figures 1 and 2, manuscript text, and model results were not changed.\n');
    catch exception
        for k = 1:numel(ownedFigures)
            if isgraphics(ownedFigures(k))
                close(ownedFigures(k));
            end
        end
        status.status = 'FAILED';
        status.error_identifier = exception.identifier;
        status.error_message = exception.message;
        status.completed_at = datestr(now, 31);
        writeJSON(fullfile(outputFolder, 'export_status.json'), status);
        fprintf(2, '\nExport stopped. Diagnostic file: %s\n', ...
            fullfile(outputFolder, 'export_status.json'));
        rethrow(exception);
    end
end

function cfg = figureSettings()
% Figure fonts, sizes and colours.
    cfg.fontName = 'Times New Roman';
    cfg.fontSize = 11;
    cfg.labelSize = 12;
    cfg.titleSize = 12;
    cfg.legendSize = 10;
    cfg.axisWidth = 0.9;
    cfg.lineWidth = 1.55;
    cfg.referenceWidth = 2.1;
    cfg.markerSize = 4.8;
    cfg.dpi = 600;
    cfg.keepFiguresOpen = false;
    cfg.saveEditableFIG = false;
    cfg.visible = 'off';
    % Shared palette for both countries.
    cfg.blue = [0.000, 0.447, 0.741];
    cfg.orange = [0.850, 0.325, 0.098];
    cfg.yellow = [0.929, 0.694, 0.125];
    cfg.purple = [0.494, 0.184, 0.556];
    cfg.green = [0.300, 0.580, 0.270];
    cfg.teal = [0.000, 0.500, 0.450];
    cfg.red = [0.720, 0.180, 0.220];
    cfg.gray = [0.400, 0.400, 0.400];
    cfg.black = [0.100, 0.100, 0.100];
    cfg.band = 0.84 * [1, 1, 1] + 0.16 * cfg.blue;
end

function map = outputNames()
% File stems are read by the existing LaTeX source; do not renumber them.
    names = {'Fig1_validation_calendar'; 'Fig2_spain_backtest'; ...
        'Fig3_france_backtest'; 'Fig4_spain_monthly'; ...
        'Fig5_france_monthly'; 'Fig6_spain_annual'; 'Fig7_france_annual'; ...
        'Fig8_spain_contributions'; 'Fig9_france_contributions'; ...
        'Fig10_spain_sensitivity'; 'Fig11_france_sensitivity'};
    topics = {'Validation calendar'; 'Spain: terminal backtest'; ...
        'France: terminal backtest'; 'Spain: monthly scenarios'; ...
        'France: monthly scenarios'; 'Spain: annual scenarios'; ...
        'France: annual scenarios'; 'Spain: signed contributions'; ...
        'France: signed contributions'; 'Spain: structural sensitivity'; ...
        'France: structural sensitivity'};
    map = table((3:13)', string(names), string(topics), ...
        'VariableNames', {'manuscript_figure', 'file_stem', 'content'});
end

function method = extractZIP(source, destination)
% Validate ZIP entries before extraction.
    entries = inspectZIP(source);
    try
        unzip(source, destination);
        verifyExtractedFiles(destination, entries);
        method = 'MATLAB unzip';
        return;
    catch nativeError
        % Some older installations use Java internally for unzip. The OS
        % fallback receives the same validated archive in a fresh folder.
        fprintf('Native ZIP extraction was unavailable. Trying the system extractor.\n');
        if isfolder(destination)
            rmdir(destination, 's');
        end
        mkdir(destination);
        [code, message] = systemArchive('extract', source, destination);
        if code ~= 0
            error('ForecastFigures:ZIPExtraction', ...
                ['The ZIP could not be extracted. MATLAB: %s\nSystem extractor: %s\n', ...
                 'You may also extract your trusted results ZIP manually and pass ', ...
                 'the extracted folder to generate_forecasting_figures.'], ...
                 nativeError.message, message);
        end
        verifyExtractedFiles(destination, entries);
        method = 'System extractor after directory validation';
    end
end

function entries = inspectZIP(source)
% Read a classic ZIP central directory; reject ZIP64 and encrypted entries.
    info = dir(source);
    if isempty(info) || info.bytes < 22 || info.bytes >= 2^32
        error('ForecastFigures:ZIPFormat', ...
            'Expected a nonempty classic ZIP smaller than 4 GiB: %s', source);
    end
    fid = fopen(source, 'rb');
    if fid < 0
        error('ForecastFigures:ZIPRead', 'Cannot open the results ZIP: %s', source);
    end
    closeFile = onCleanup(@() fclose(fid)); %#ok<NASGU>
    tailLength = min(info.bytes, 65557);
    fseek(fid, info.bytes-tailLength, 'bof');
    tail = fread(fid, tailLength, '*uint8')';
    candidates = find(tail(1:end-3) == 80 & tail(2:end-2) == 75 & ...
        tail(3:end-1) == 5 & tail(4:end) == 6);
    eocd = [];
    for k = numel(candidates):-1:1
        pos = candidates(k);
        if pos+21 <= numel(tail) && ...
                pos+21+zipNumber(tail,pos+20,2) == numel(tail)
            eocd = pos;
            break;
        end
    end
    if isempty(eocd)
        error('ForecastFigures:ZIPFormat', 'The ZIP end-of-directory record is missing or truncated.');
    end
    disk = zipNumber(tail,eocd+4,2);
    directoryDisk = zipNumber(tail,eocd+6,2);
    diskCount = zipNumber(tail,eocd+8,2);
    count = zipNumber(tail,eocd+10,2);
    directorySize = zipNumber(tail,eocd+12,4);
    directoryOffset = zipNumber(tail,eocd+16,4);
    endOffset = info.bytes-tailLength+eocd-1;
    if disk ~= 0 || directoryDisk ~= 0 || diskCount ~= count || ...
            count == 65535 || directorySize == 2^32-1 || directoryOffset == 2^32-1
        error('ForecastFigures:ZIPFormat', ...
            'Split and ZIP64 archives are not supported by this directory reader.');
    end
    if count < 1 || count > 50000 || directorySize > 64*1024^2 || ...
            directoryOffset+directorySize ~= endOffset
        error('ForecastFigures:ZIPFormat', 'The ZIP directory size or position is not valid.');
    end
    fseek(fid, directoryOffset, 'bof');
    directory = fread(fid, directorySize, '*uint8')';
    if numel(directory) ~= directorySize
        error('ForecastFigures:ZIPRead', 'The ZIP directory is incomplete.');
    end
    entries = repmat(struct('name','','bytes',0,'is_directory',false),count,1);
    fileRanges = zeros(count,2);
    position = 1;
    totalBytes = 0;
    for k = 1:count
        if position+45 > directorySize || ...
                zipNumber(directory,position,4) ~= hex2dec('02014B50')
            error('ForecastFigures:ZIPFormat', 'Invalid ZIP directory entry %d.', k);
        end
        h = directory(position:position+45);
        flags = zipNumber(h,9,2);
        compression = zipNumber(h,11,2);
        compressedBytes = zipNumber(h,21,4);
        bytes = zipNumber(h,25,4);
        nameLength = zipNumber(h,29,2);
        extraLength = zipNumber(h,31,2);
        commentLength = zipNumber(h,33,2);
        startDisk = zipNumber(h,35,2);
        externalAttributes = zipNumber(h,39,4);
        localOffset = zipNumber(h,43,4);
        entryEnd = position+46+nameLength+extraLength+commentLength-1;
        if entryEnd > directorySize || nameLength < 1 || ...
                any([bytes compressedBytes localOffset] == 2^32-1) || startDisk ~= 0
            error('ForecastFigures:ZIPFormat', 'Unsupported or truncated ZIP entry %d.', k);
        end
        if bitand(uint16(flags),uint16(1)) ~= 0 || ~ismember(compression,[0 8])
            error('ForecastFigures:ZIPFormat', ...
                'Only unencrypted stored/deflated ZIP entries are supported.');
        end
        rawName = directory(position+46:position+45+nameLength);
        if bitand(uint16(flags),uint16(2048)) ~= 0
            name = native2unicode(rawName,'UTF-8');
            if ~isequal(unicode2native(name,'UTF-8'),rawName)
                error('ForecastFigures:ZIPFormat', 'Invalid UTF-8 ZIP entry name.');
            end
        elseif any(rawName > 127)
            error('ForecastFigures:ZIPEncoding', ...
                'A non-ASCII entry name has no UTF-8 flag. Extract this trusted archive manually.');
        else
            name = char(rawName);
        end
        isDirectory = name(end) == '/';
        validateZIPName(name);
        unixType = bitand(bitshift(uint32(externalAttributes),-16),uint32(61440));
        if ~ismember(double(unixType),[0 16384 32768]) || ...
                (unixType == 16384 && ~isDirectory) || ...
                (unixType == 32768 && isDirectory)
            error('ForecastFigures:UnsafeZIP', ...
                'Links or special files are not accepted in the archive: %s', name);
        end
        totalBytes = totalBytes+bytes;
        if bytes > 512*1024^2 || totalBytes > 2*1024^3 || ...
                (isDirectory && bytes ~= 0)
            error('ForecastFigures:ZIPSize', 'The archive exceeds the plotting reader size limits.');
        end
        % A second filename lives in each local header. Check it as well so
        % the name validated here is the name the extractor will encounter.
        if localOffset+30 > directoryOffset
            error('ForecastFigures:ZIPFormat', 'Invalid local header offset.');
        end
        fseek(fid,localOffset,'bof');
        local = fread(fid,30,'*uint8')';
        if numel(local) ~= 30 || zipNumber(local,1,4) ~= hex2dec('04034B50')
            error('ForecastFigures:ZIPFormat', 'Invalid local header for %s.',name);
        end
        localNameLength = zipNumber(local,27,2);
        localExtraLength = zipNumber(local,29,2);
        localName = fread(fid,localNameLength,'*uint8')';
        dataEnd = localOffset+30+localNameLength+localExtraLength+compressedBytes;
        if ~isequal(localName,rawName) || zipNumber(local,7,2) ~= flags || ...
                zipNumber(local,9,2) ~= compression || dataEnd > directoryOffset
            error('ForecastFigures:ZIPFormat', 'Local and directory entries disagree for %s.',name);
        end
        if bitand(uint16(flags),uint16(8)) == 0 && ...
                (zipNumber(local,15,4) ~= zipNumber(h,17,4) || ...
                 zipNumber(local,19,4) ~= compressedBytes || zipNumber(local,23,4) ~= bytes)
            error('ForecastFigures:ZIPFormat', 'Local sizes/checksum disagree for %s.',name);
        end
        entries(k) = struct('name',name,'bytes',bytes,'is_directory',isDirectory);
        fileRanges(k,:) = [localOffset,dataEnd];
        position = entryEnd+1;
    end
    if position ~= directorySize+1
        error('ForecastFigures:ZIPFormat', 'Unexpected extra records in the ZIP directory.');
    end
    keys = cellfun(@(n) lower(regexprep(n,'/$','')), {entries.name}, 'UniformOutput',false);
    if numel(unique(keys)) ~= count
        error('ForecastFigures:UnsafeZIP', 'Duplicate or case-colliding ZIP entry names were found.');
    end
    fileKeys = keys(~[entries.is_directory]);
    for k = 1:count
        parts = regexp(keys{k},'/', 'split');
        for j = 1:numel(parts)-1
            if any(strcmp(strjoin(parts(1:j),'/'),fileKeys))
                error('ForecastFigures:UnsafeZIP', 'A ZIP file is also used as a directory.');
            end
        end
    end
    fileRanges = sortrows(fileRanges,1);
    if any(fileRanges(2:end,1) < fileRanges(1:end-1,2))
        error('ForecastFigures:ZIPFormat', 'ZIP entries have overlapping storage ranges.');
    end
end

function number = zipNumber(bytes,first,width)
% ZIP integers are little-endian, independently of the host computer.
    if first < 1 || first+width-1 > numel(bytes)
        error('ForecastFigures:ZIPRead', 'An incomplete ZIP field was encountered.');
    end
    number = sum(double(bytes(first:first+width-1)).*256.^(0:width-1));
end

function validateZIPName(name)
% Keep extraction inside its temporary folder on Windows, macOS and Linux.
    if isempty(name) || name(1) == '/' || any(name == '\') || ...
            any(double(name) < 32) || any(double(name) == 127) || ...
            ~isempty(regexp(name,'[<>:"|?*]','once'))
        error('ForecastFigures:UnsafeZIP', 'Unsafe ZIP entry name: %s',name);
    end
    parts = regexp(regexprep(name,'/$',''),'/', 'split');
    for j = 1:numel(parts)
        part = parts{j};
        if isempty(part) || strcmp(part,'.') || strcmp(part,'..') || ...
                ismember(part(end),[' ' '.']) || ...
                ~isempty(regexpi(part,'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\.|$)','once'))
            error('ForecastFigures:UnsafeZIP', 'Unsafe ZIP path: %s',name);
        end
    end
end

function verifyExtractedFiles(destination,entries)
    for k = 1:numel(entries)
        target = fullfile(destination,strrep(entries(k).name,'/',filesep));
        if entries(k).is_directory
            exists = isfolder(target);
        else
            info = dir(target);
            exists = numel(info) == 1 && ~info.isdir && info.bytes == entries(k).bytes;
        end
        if ~exists
            error('ForecastFigures:IncompleteExtraction', ...
                'An extracted entry is missing or has the wrong size: %s',entries(k).name);
        end
    end
end

function [method,message] = createFigureZIP(outputFolder)
% A packaging failure must not discard successfully exported PDF/PNG files.
    bundle = fullfile(outputFolder,'Figures_for_Overleaf.zip');
    message = '';
    try
        zip(bundle, {'Figures'}, outputFolder);
        inspectZIP(bundle);
        method = 'MATLAB zip';
        return;
    catch nativeError
        if isfile(bundle)
            delete(bundle);
        end
        [code,systemMessage] = systemArchive('create',bundle,outputFolder);
        if code == 0 && isfile(bundle)
            try
                inspectZIP(bundle);
                method = 'System ZIP utility';
                return;
            catch systemError
                systemMessage = systemError.message;
            end
        end
        if isfile(bundle)
            delete(bundle);
        end
        method = 'not_created';
        message = sprintf(['All figures were exported, but the convenience ZIP could not be created. ', ...
            'Upload the files in Figures/ directly. MATLAB: %s. System: %s'], ...
            nativeError.message,systemMessage);
    end
end

function [code,message] = systemArchive(action,archive,folder)
% Pass paths in environment variables rather than inserting them as code.
% The variables are restored even when a command fails.
    names = {'FORECAST_FIGURES_ARCHIVE','FORECAST_FIGURES_DIRECTORY'};
    oldValues = {getenv(names{1}),getenv(names{2})};
    restore = onCleanup(@() restoreArchiveEnvironment(names,oldValues)); %#ok<NASGU>
    setenv(names{1},archive);
    setenv(names{2},folder);
    if ispc
        if strcmp(action,'extract')
            command = ['powershell.exe -NoLogo -NoProfile -NonInteractive -Command ', ...
                '"$ErrorActionPreference = ''Stop''; Expand-Archive ', ...
                '-LiteralPath $env:FORECAST_FIGURES_ARCHIVE ', ...
                '-DestinationPath $env:FORECAST_FIGURES_DIRECTORY -Force"'];
        else
            command = ['powershell.exe -NoLogo -NoProfile -NonInteractive -Command ', ...
                '"$ErrorActionPreference = ''Stop''; Compress-Archive ', ...
                '-LiteralPath (Join-Path $env:FORECAST_FIGURES_DIRECTORY ''Figures'') ', ...
                '-DestinationPath $env:FORECAST_FIGURES_ARCHIVE -Force"'];
        end
    elseif isunix
        if strcmp(action,'extract')
            command = ['unzip -q -o "$FORECAST_FIGURES_ARCHIVE" ', ...
                '-d "$FORECAST_FIGURES_DIRECTORY"'];
        else
            command = ['cd "$FORECAST_FIGURES_DIRECTORY" && ', ...
                'zip -q -r "$FORECAST_FIGURES_ARCHIVE" Figures'];
        end
    else
        code = -1;
        message = 'No system ZIP fallback is configured for this operating system.';
        return;
    end
    [code,message] = system(command);
end

function restoreArchiveEnvironment(names,values)
    for k = 1:numel(names)
        setenv(names{k},values{k});
    end
end

function root = locateRun(extractedFolder)
    candidates = dir(fullfile(extractedFolder, '**', 'monthly_peak_scenarios.csv'));
    roots = {};
    for k = 1:numel(candidates)
        [countryFolder, scenario] = fileparts(candidates(k).folder);
        [candidateRoot, country] = fileparts(countryFolder);
        if strcmpi(scenario, 'TURNOVER') && strcmpi(country, 'spain') ...
                && isfile(fullfile(candidateRoot, 'france', 'TURNOVER', 'monthly_peak_scenarios.csv'))
            roots{end + 1} = candidateRoot; %#ok<AGROW>
        end
    end
    roots = unique(roots, 'stable');
    if isempty(roots)
        error('ForecastFigures:WrongArchive', ...
            ['This ZIP does not contain the physical-scenario results structure. ', ...
             'Expected spain/TURNOVER/monthly_peak_scenarios.csv and the ', ...
             'corresponding france folder. Select the results ZIP, not the Overleaf ZIP.']);
    elseif numel(roots) == 1
        root = roots{1};
    else
        [choice, ok] = listdlg('PromptString', 'Select one results run:', ...
            'SelectionMode', 'single', 'ListString', roots, 'ListSize', [650 220]);
        if ~ok
            error('ForecastFigures:NoRunSelected', 'No results run was selected.');
        end
        root = roots{choice};
    end
end

function verifyRunStatus(root)
    file = fullfile(root, 'execution_status.json');
    requireFile(file);
    state = jsondecode(fileread(file));
    if ~isfield(state, 'status') || ~startsWith(upper(string(state.status)), 'COMPLETED')
        error('ForecastFigures:IncompleteRun', 'The selected experiment is not marked as completed.');
    end
    if isfield(state, 'all_checks_passed') && ~state.all_checks_passed
        error('ForecastFigures:FailedSourceChecks', 'The source experiment reports failed checks.');
    end
end

function [D, records] = readCountryData(root, country)
    base = fullfile(root, country);
    records = struct('role', {}, 'path', {}, 'rows', {});
    benchmarkFile = fullfile(base, 'statistical_benchmark_paths.csv');
    B = readCSV(benchmarkFile);
    requireColumns(B, {'phase','origin','ds','horizon','model','y_true','pred_MW'}, benchmarkFile);
    records(end + 1) = sourceRecord('calendar_and_backtest', benchmarkFile, height(B));
    D.country = country;
    D.benchmark = B;
    D.calendar = calendarFromPaths(B);
    D.holdout = holdoutFromPaths(B);
    [D.history, historyFile] = observedHistory(country, root, B);
    records(end + 1) = sourceRecord('observed_monthly_history', historyFile, height(D.history));

    cases = {'BASELINE_OVERLAP100', 'TURNOVER', 'MILESTONES', 'FAST_RENEWAL'};
    for k = 1:numel(cases)
        name = cases{k};
        monthlyFile = fullfile(base, name, 'monthly_peak_scenarios.csv');
        annualFile = fullfile(base, name, 'annual_peak_scenarios.csv');
        M = loadMonthly(monthlyFile);
        A = loadAnnual(annualFile);
        checkAnnualMaximum(M, A, name);
        D.monthly.(name) = M;
        D.annual.(name) = A;
        records(end + 1) = sourceRecord(['monthly_' name], monthlyFile, height(M));
        records(end + 1) = sourceRecord(['annual_' name], annualFile, height(A));
    end
    baseM = D.monthly.BASELINE_OVERLAP100;
    refM = D.monthly.TURNOVER;
    assertNear(baseM.structural_increment_MW, zeros(height(baseM),1), ...
        'The zero-additionality scenario does not reproduce the statistical baseline.');
    for k = 1:numel(cases)
        M = D.monthly.(cases{k});
        if ~isequal(M.ds, baseM.ds) || ~isequal(D.annual.(cases{k}).year, D.annual.BASELINE_OVERLAP100.year)
            error('ForecastFigures:MisalignedScenario', 'Scenario calendars differ for %s.', country);
        end
    end
    for field = {'P5_MW','P50_MW','P95_MW','deterministic_MW'}
        f = field{1};
        assertNear(refM.(f) - baseM.(f), refM.structural_increment_MW, ...
            ['The fixed-increment identity failed for ' country '/' f '.']);
    end
    if dateshift(D.history.ds(end), 'start', 'month') + calmonths(1) ~= refM.ds(1)
        error('ForecastFigures:HistoryBoundary', 'Observed and projected monthly calendars are not consecutive.');
    end

    contributionFile = fullfile(base, 'TURNOVER', 'annual_deterministic_contributions.csv');
    C = readCSV(contributionFile);
    requireColumns(C, {'year','peak_month','total_increment_MW', ...
        'total_deterministic_peak_MW','baseline_at_selected_month_MW'}, contributionFile);
    C.year = numericColumn(C, 'year');
    C.peak_month = dateColumn(C, 'peak_month');
    keyNames = C.Properties.VariableNames;
    contributionNames = keyNames(endsWith(keyNames, '_increment_MW') ...
        & ~strcmp(keyNames, 'total_increment_MW'));
    if isempty(contributionNames)
        error('ForecastFigures:NoContributions', 'No component columns were found for %s.', country);
    end
    components = zeros(height(C), numel(contributionNames));
    for k = 1:numel(contributionNames)
        C.(contributionNames{k}) = numericColumn(C, contributionNames{k});
        components(:,k) = C.(contributionNames{k});
    end
    for field = {'total_increment_MW','total_deterministic_peak_MW','baseline_at_selected_month_MW'}
        C.(field{1}) = numericColumn(C, field{1});
    end
    C = sortrows(C, 'year');
    % Summing signed terms must reproduce the reported total, not its P50.
    assertNear(sum(components,2), numericColumn(readCSV(contributionFile),'total_increment_MW'), ...
        ['Signed components do not add up for ' country '.']);
    assertNear(C.baseline_at_selected_month_MW + C.total_increment_MW, ...
        C.total_deterministic_peak_MW, 'The deterministic accounting identity failed.');
    if ~isequal(C.year, D.annual.TURNOVER.year)
        error('ForecastFigures:ContributionYears', 'Contribution years do not match annual results.');
    end
    assertNear(C.total_deterministic_peak_MW, D.annual.TURNOVER.deterministic_max_MW, ...
        'The contribution table is not aligned with the annual deterministic maximum.');
    [found, monthIndex] = ismember(C.peak_month, refM.ds);
    if ~all(found)
        error('ForecastFigures:PeakMonthMissing', 'A selected peak month is missing from monthly results.');
    end
    assertNear(refM.deterministic_MW(monthIndex), C.total_deterministic_peak_MW, ...
        'A contribution table points to the wrong peak month.');
    D.contributions = C;
    records(end + 1) = sourceRecord('signed_contributions', contributionFile, height(C));

    [caseIDs, labels] = sensitivityNames();
    S = table(string(caseIDs(:)), string(labels(:)), zeros(numel(caseIDs),1), ...
        'VariableNames', {'scenario','description','P50_MW'});
    terminalYear = D.annual.TURNOVER.year(end);
    for k = 1:numel(caseIDs)
        file = fullfile(base, caseIDs{k}, 'annual_peak_scenarios.csv');
        A = loadAnnual(file);
        row = A.year == terminalYear;
        if nnz(row) ~= 1
            error('ForecastFigures:SensitivityYear', 'Missing or duplicated terminal year in %s.', file);
        end
        S.P50_MW(k) = A.P50_MW(row);
        records(end + 1) = sourceRecord(['sensitivity_' caseIDs{k}], file, height(A));
    end
    D.sensitivity = S;
    D.terminalYear = terminalYear;
    fprintf('Validated %s: %d observed months, %d forecast months, %d annual rows.\n', ...
        country, height(D.history), height(refM), height(D.annual.TURNOVER));
end

function record = sourceRecord(role, path, rows)
    record = struct('role', role, 'path', path, 'rows', rows);
end

function T = readCSV(file)
    requireFile(file);
    options = detectImportOptions(file, 'Delimiter', ',');
    options.VariableNamingRule = 'preserve';
    % Explicit text import prevents locale-dependent date guesses.
    options = setvartype(options, options.VariableNames, 'string');
    T = readtable(file, options);
    if isempty(T)
        error('ForecastFigures:EmptyCSV', 'The input table is empty: %s', file);
    end
end

function requireFile(file)
    if ~isfile(file)
        error('ForecastFigures:MissingInput', 'Required input not found: %s', file);
    end
end

function requireColumns(T, columns, file)
    absent = columns(~ismember(columns, T.Properties.VariableNames));
    if ~isempty(absent)
        error('ForecastFigures:MissingColumns', 'Missing columns in %s: %s', ...
            file, strjoin(absent, ', '));
    end
end

function x = numericColumn(T, name)
    requireColumns(T, {name}, 'input table');
    value = T.(name);
    if isnumeric(value)
        x = double(value);
    else
        x = str2double(string(value));
    end
    x = x(:);
    if any(~isfinite(x))
        error('ForecastFigures:NonfiniteData', 'Column %s contains a missing or nonnumeric value.', name);
    end
end

function d = dateColumn(T, name)
    requireColumns(T, {name}, 'input table');
    if isdatetime(T.(name))
        d = T.(name);
    else
        text = string(T.(name));
        if any(strlength(text) < 10)
            error('ForecastFigures:InvalidDate', 'Expected ISO dates in column %s.', name);
        end
        d = datetime(extractBetween(text, 1, 10), 'InputFormat', 'yyyy-MM-dd');
    end
    d = d(:);
    d.Format = 'yyyy-MM-dd';
    if any(isnat(d))
        error('ForecastFigures:InvalidDate', 'Missing dates in column %s.', name);
    end
end

function M = loadMonthly(file)
    T = readCSV(file);
    names = {'deterministic_MW','P5_MW','P50_MW','P95_MW','structural_increment_MW'};
    requireColumns(T, [{'ds'},names], file);
    M = table(dateColumn(T,'ds'), 'VariableNames', {'ds'});
    for k = 1:numel(names)
        M.(names{k}) = numericColumn(T,names{k});
    end
    M = sortrows(M,'ds');
    if numel(unique(M.ds)) ~= height(M) || ...
            any(M.ds(2:end) ~= M.ds(1:end-1) + calmonths(1))
        error('ForecastFigures:MonthlyCalendar', 'Duplicated or missing monthly dates in %s.', file);
    end
    if any(day(M.ds) ~= 1)
        error('ForecastFigures:MonthStamp', 'Expected first-of-month timestamps in %s.', file);
    end
    validateQuantiles(M, file);
end

function A = loadAnnual(file)
    T = readCSV(file);
    names = {'year','P5_MW','P50_MW','P95_MW','deterministic_max_MW','n_simulations'};
    requireColumns(T,names,file);
    A = table;
    for k = 1:numel(names)
        A.(names{k}) = numericColumn(T,names{k});
    end
    A = sortrows(A,'year');
    if any(diff(A.year) ~= 1) || any(A.year ~= floor(A.year))
        error('ForecastFigures:AnnualCalendar', 'Missing or duplicated annual dates in %s.', file);
    end
    if any(A.n_simulations ~= 5000)
        error('ForecastFigures:SimulationCount', 'This manuscript exporter expects 5000 simulations in %s.', file);
    end
    validateQuantiles(A, file);
end

function validateQuantiles(T, file)
    if any(T.P5_MW > T.P50_MW + 1e-6 | T.P50_MW > T.P95_MW + 1e-6)
        error('ForecastFigures:QuantileOrder', 'Quantiles are not ordered in %s.', file);
    end
    if any(T.P5_MW <= 0)
        error('ForecastFigures:NonpositiveDemand', 'A nonpositive demand quantile was found in %s.', file);
    end
end

function checkAnnualMaximum(M, A, label)
% Only the deterministic maximum can be recovered from monthly CSV columns.
% Annual percentiles are read as exported, never as maxima of monthly P50s.
    expectedYears = unique(year(M.ds));
    if ~isequal(expectedYears, A.year)
        error('ForecastFigures:YearAlignment', 'Monthly and annual calendars differ in %s.', label);
    end
    for k = 1:height(A)
        rows = year(M.ds) == A.year(k);
        if nnz(rows) ~= 12
            error('ForecastFigures:PartialYear', 'Year %d does not contain 12 months.', A.year(k));
        end
        assertNear(max(M.deterministic_MW(rows)), A.deterministic_max_MW(k), ...
            ['Annual deterministic maximum differs in ' label '.']);
    end
end

function assertNear(a,b,message)
    if ~isequal(size(a),size(b)) || any(abs(a(:)-b(:)) > 1e-6)
        error('ForecastFigures:NumericalCheck', '%s', message);
    end
end

function C = calendarFromPaths(B)
    rows = string(B.model) == "ENSEMBLE";
    T = B(rows,:);
    if isempty(T)
        error('ForecastFigures:MissingEnsemble', 'No ENSEMBLE paths were found.');
    end
    origin = dateColumn(T,'origin');
    dates = dateColumn(T,'ds');
    horizons = numericColumn(T,'horizon');
    origins = unique(origin);
    starts = NaT(size(origins));
    ends = NaT(size(origins));
    phase = strings(size(origins));
    for k = 1:numel(origins)
        r = origin == origins(k);
        pathDates = sort(dates(r));
        if numel(pathDates) ~= 24 || ~isequal(sort(horizons(r)), (1:24)')
            error('ForecastFigures:IncompletePath', 'Every calendar path must contain horizons 1--24.');
        end
        if ~isequal(pathDates, origins(k) + calmonths((1:24)'))
            error('ForecastFigures:PathDates', 'Forecast horizons do not match their calendar dates.');
        end
        p = unique(string(T.phase(r)));
        if numel(p) ~= 1
            error('ForecastFigures:AmbiguousPhase', 'An origin has more than one phase.');
        end
        phase(k) = p;
        starts(k) = pathDates(1);
        ends(k) = pathDates(end);
    end
    allowed = ["weight_calibration","probability_tuning","terminal_holdout"];
    if ~all(ismember(phase,allowed))
        error('ForecastFigures:UnknownPhase', 'Unknown chronological roles in the source archive.');
    end
    C = table(origins,starts,ends,phase, ...
        'VariableNames', {'origin','target_start','target_end','phase'});
end

function H = holdoutFromPaths(B)
    ids = {'LSTM','LGBM','TFT','ENSEMBLE'};
    T = B(string(B.phase) == "terminal_holdout",:);
    if numel(unique(string(T.origin))) ~= 1
        error('ForecastFigures:TerminalOrigins', 'Expected exactly one terminal origin.');
    end
    H = table;
    for k = 1:numel(ids)
        R = T(string(T.model) == string(ids{k}),:);
        d = dateColumn(R,'ds');
        [d,order] = sort(d);
        y = numericColumn(R,'y_true'); y = y(order);
        p = numericColumn(R,'pred_MW'); p = p(order);
        if numel(d) ~= 24 || numel(unique(d)) ~= 24
            error('ForecastFigures:TerminalLength', 'Missing or duplicated terminal model predictions.');
        end
        if k == 1
            H.ds = d;
            H.observed_MW = y;
        else
            if ~isequal(H.ds,d)
                error('ForecastFigures:TerminalAlignment', 'Terminal model dates do not agree.');
            end
            assertNear(H.observed_MW,y,'Terminal observed targets differ between models.');
        end
        H.(ids{k}) = p;
    end
end

function writePlotData(D, folder, country)
% Save the exact plotted values, preserving MW in their source columns.
    writetable(D.history,fullfile(folder,[country '_observed_monthly_MW.csv']));
    writetable(D.calendar,fullfile(folder,[country '_validation_calendar.csv']));
    writetable(D.holdout,fullfile(folder,[country '_terminal_model_paths_MW.csv']));
    writetable(D.monthly.TURNOVER,fullfile(folder,[country '_turnover_monthly_MW.csv']));
    writetable(D.monthly.BASELINE_OVERLAP100,fullfile(folder,[country '_baseline_monthly_MW.csv']));
    names = fieldnames(D.annual);
    for k = 1:numel(names)
        writetable(D.annual.(names{k}),fullfile(folder,[country '_' names{k} '_annual_MW.csv']));
    end
    writetable(D.contributions,fullfile(folder,[country '_signed_contributions_MW.csv']));
    writetable(D.sensitivity,fullfile(folder,[country '_selected_sensitivities_MW.csv']));
end

function [fig,ax] = makeAxes(cfg,kind)
    dimensions = [18.5 10.8];
    position = [0.095 0.235 0.875 0.66];
    if strcmp(kind,'calendar')
        dimensions = [19.5 12.0];
        position = [0.13 0.22 0.84 0.68];
    elseif strcmp(kind,'contributions')
        dimensions = [19.5 12.0];
        position = [0.105 0.28 0.865 0.62];
    elseif strcmp(kind,'sensitivity')
        dimensions = [23.0 13.5];
        position = [0.355 0.13 0.61 0.775];
    end
    fig = figure('Visible',cfg.visible,'Color','w','Units','centimeters', ...
        'Position',[2 2 dimensions],'PaperPositionMode','auto', ...
        'InvertHardcopy','off','NumberTitle','off','MenuBar','none','ToolBar','none');
    ax = axes('Parent',fig,'Units','normalized','Position',position, ...
        'FontName',cfg.fontName,'FontSize',cfg.fontSize,'FontWeight','normal', ...
        'LineWidth',cfg.axisWidth,'Box','on','TickDir','in','TickLength',[0.012 0.025], ...
        'XColor',cfg.black,'YColor',cfg.black,'Color','w', ...
        'XGrid','on','YGrid','on','GridColor',[0.82 0.82 0.82], ...
        'GridAlpha',0.5,'GridLineStyle',':','Layer','top', ...
        'TickLabelInterpreter','none');
    hold(ax,'on');
    if isprop(ax,'Toolbar') && ~isempty(ax.Toolbar)
        ax.Toolbar.Visible = 'off';
    end
    ax.XAxis.Exponent = 0;
    ax.YAxis.Exponent = 0;
end

function labelAxes(ax,cfg,xtext,ytext,ttext)
    xlabel(ax,xtext,'FontName',cfg.fontName,'FontSize',cfg.labelSize,'Interpreter','none');
    ylabel(ax,ytext,'FontName',cfg.fontName,'FontSize',cfg.labelSize,'Interpreter','none');
    title(ax,ttext,'FontName',cfg.fontName,'FontSize',cfg.titleSize, ...
        'FontWeight','normal','Interpreter','none');
end

function legendBelow(ax,handles,labels,columns,cfg)
    legend(ax,handles,labels,'Location','southoutside','Orientation','horizontal', ...
        'NumColumns',columns,'FontName',cfg.fontName,'FontSize',cfg.legendSize, ...
        'Interpreter','none','Box','off','AutoUpdate','off');
end

function h = drawBand(ax,x,low,high,cfg)
% An opaque light fill keeps the PDF vector-based without alpha blending.
    h = patch(ax,[x(:);flipud(x(:))],[low(:);flipud(high(:))],cfg.band, ...
        'EdgeColor','none','FaceAlpha',1);
end

function fig = plotCalendar(C,cfg)
    [fig,ax] = makeAxes(cfg,'calendar');
    colors = [cfg.blue;cfg.orange;cfg.teal];
    markers = {'o','s','d'};
    names = {'Weight calibration','Prequential probability audit','Terminal retrospective evaluation'};
    phases = ["weight_calibration","probability_tuning","terminal_holdout"];
    handles = gobjects(3,1);
    for k = 1:height(C)
        group = find(C.phase(k) == phases,1);
        x = datenum([C.target_start(k);C.target_end(k)]);
        plot(ax,x,[k k],'-','Color',colors(group,:),'LineWidth',1.9,'HandleVisibility','off');
        plot(ax,x(1),k,markers{group},'Color',colors(group,:), ...
            'MarkerFaceColor',colors(group,:),'MarkerSize',cfg.markerSize,'HandleVisibility','off');
    end
    for k = 1:3
        handles(k) = plot(ax,nan,nan,['-' markers{k}],'Color',colors(k,:), ...
            'LineWidth',1.9,'MarkerFaceColor',colors(k,:),'MarkerSize',cfg.markerSize);
    end
    ax.YDir = 'reverse';
    ylim(ax,[0.5 height(C)+0.5]);
    yticks(ax,1:height(C));
    yticklabels(ax,cellstr(datestr(C.origin,'yyyy-mm')));
    years = (year(min(C.target_start)) + 1):year(max(C.target_end)) + 1;
    xticks(ax,datenum(years,ones(size(years)),ones(size(years))));
    xticklabels(ax,cellstr(string(years)));
    xlim(ax,[datenum(min(C.target_start))-110,datenum(max(C.target_end))+65]);
    ax.YGrid = 'off';
    labelAxes(ax,cfg,'Target month','Forecast origin','Chronological evaluation of the statistical baseline');
    legendBelow(ax,handles,names,2,cfg);
end

function fig = plotBacktest(D,name,cfg)
    [fig,ax] = makeAxes(cfg,'standard');
    H = D.holdout;
    x = datenum(H.ds);
    h = gobjects(5,1);
    h(1) = plot(ax,x,H.observed_MW/1000,'-o','Color',cfg.black, ...
        'LineWidth',1.7,'MarkerSize',cfg.markerSize,'MarkerFaceColor','w');
    h(2) = plot(ax,x,H.LSTM/1000,':','Color',cfg.orange,'LineWidth',cfg.lineWidth);
    h(3) = plot(ax,x,H.LGBM/1000,'--','Color',cfg.blue,'LineWidth',cfg.lineWidth);
    h(4) = plot(ax,x,H.TFT/1000,'-.','Color',cfg.purple,'LineWidth',cfg.lineWidth);
    h(5) = plot(ax,x,H.ENSEMBLE/1000,'-','Color',cfg.teal,'LineWidth',cfg.referenceWidth);
    ticks = H.ds(1) + calmonths(0:3:24);
    xticks(ax,datenum(ticks));
    xticklabels(ax,cellstr(datestr(ticks,'mmm yyyy')));
    xlim(ax,[x(1)-20,datenum(ticks(end))+10]);
    setPaddedY(ax,[H.observed_MW;H.LSTM;H.LGBM;H.TFT;H.ENSEMBLE]/1000);
    origin = H.ds(1)-calmonths(1);
    labelAxes(ax,cfg,'Target month','Monthly peak demand [GW]', ...
        [name ': recursive forecast issued in ' datestr(origin,'mmmm yyyy')]);
    legendBelow(ax,h,{'Observed','LSTM','LightGBM','TFT','Ensemble'},5,cfg);
end

function fig = plotMonthly(D,name,cfg)
    [fig,ax] = makeAxes(cfg,'standard');
    M = D.monthly.TURNOVER;
    B = D.monthly.BASELINE_OVERLAP100;
    x = datenum(M.ds);
    band = drawBand(ax,x,M.P5_MW/1000,M.P95_MW/1000,cfg);
    observed = plot(ax,datenum(D.history.ds),D.history.y_mw/1000, ...
        '-','Color',cfg.gray,'LineWidth',1.05);
    base = plot(ax,x,B.P50_MW/1000,'--','Color',cfg.orange,'LineWidth',1.10);
    reference = plot(ax,x,M.P50_MW/1000,'-','Color',cfg.blue,'LineWidth',1.35);
    setPaddedY(ax,[D.history.y_mw;M.P5_MW;M.P95_MW;B.P50_MW]/1000);
    xline(ax,x(1),':','Color',[0.45 0.45 0.45],'LineWidth',0.85,'HandleVisibility','off');
    firstYear = year(D.history.ds(1));
    finalYear = year(M.ds(end));
    years = ceil(firstYear/5)*5:5:finalYear;
    xticks(ax,datenum(years,ones(size(years)),ones(size(years))));
    xticklabels(ax,cellstr(string(years)));
    xlim(ax,[datenum(D.history.ds(1))-15,x(end)+31]);
    labelAxes(ax,cfg,'Calendar year','Monthly peak demand [GW]', ...
        [name ': observed peaks and conditional monthly scenarios']);
    rangeText = ['Conditional P5' char(8211) 'P95'];
    legendBelow(ax,[reference,band,observed,base], ...
        {'Turnover scenario: P50',rangeText,'Observed monthly peaks','Statistical baseline: P50'},2,cfg);
end

function fig = plotAnnual(D,name,cfg)
    [fig,ax] = makeAxes(cfg,'standard');
    T = D.annual.TURNOVER;
    B = D.annual.BASELINE_OVERLAP100;
    M = D.annual.MILESTONES;
    F = D.annual.FAST_RENEWAL;
    band = drawBand(ax,T.year,T.P5_MW/1000,T.P95_MW/1000,cfg);
    baseline = plot(ax,B.year,B.P50_MW/1000,':','Color',cfg.gray,'LineWidth',1.6);
    milestones = plot(ax,M.year,M.P50_MW/1000,'--s','Color',cfg.orange, ...
        'LineWidth',1.55,'MarkerSize',4,'MarkerIndices',1:3:height(M),'MarkerFaceColor','w');
    fast = plot(ax,F.year,F.P50_MW/1000,'-.^','Color',cfg.green, ...
        'LineWidth',1.6,'MarkerSize',4.2,'MarkerIndices',2:3:height(F),'MarkerFaceColor','w');
    reference = plot(ax,T.year,T.P50_MW/1000,'-o','Color',cfg.blue, ...
        'LineWidth',cfg.referenceWidth,'MarkerSize',4,'MarkerFaceColor',cfg.blue);
    xlim(ax,[T.year(1)-0.4,T.year(end)+0.4]);
    ticks = unique([T.year(1),ceil(T.year(1)/5)*5:5:T.year(end)]);
    xticks(ax,ticks);
    setPaddedY(ax,[T.P5_MW;T.P95_MW;B.P50_MW;M.P50_MW;F.P50_MW]/1000);
    labelAxes(ax,cfg,'Calendar year','Annual peak demand [GW]', ...
        [name ': conditional annual peak-demand scenarios']);
    rangeText = ['Turnover: conditional P5' char(8211) 'P95'];
    legendBelow(ax,[reference,band,milestones,fast,baseline], ...
        {'Turnover: P50',rangeText,'Milestones: P50','Fast renewal: P50','Statistical baseline: P50'},2,cfg);
end

function fig = plotContributions(D,name,cfg)
    [fig,ax] = makeAxes(cfg,'contributions');
    C = D.contributions;
    selectedYears = [2030;2035;2045];
    [present,row] = ismember(selectedYears,C.year);
    if ~all(present)
        error('ForecastFigures:ContributionHorizons', 'The manuscript requires contribution years 2030, 2035 and 2045.');
    end
    C = C(row,:);
    keys = {'EV_BEV_increment_MW','EV_PHEV_increment_MW', ...
        'HP_electricity_new_increment_MW','HP_resistance_removed_increment_MW', ...
        'industry_increment_MW','electrolyzers_increment_MW', ...
        'data_centers_increment_MW','ports_increment_MW','rail_increment_MW'};
    labels = {'Battery electric cars','Plug-in hybrid cars','New heat pumps', ...
        'Displaced resistance heating','Industry','Electrolyzers','Data centers','Ports','Rail'};
    colors = [cfg.blue;cfg.yellow;cfg.green;0.82 0.40 0.42; ...
        cfg.purple;cfg.orange;cfg.teal;0.48 0.34 0.24;0.58 0.66 0.72];
    x = (1:3)';
    positive = zeros(3,1);
    negative = zeros(3,1);
    handles = gobjects(0);
    legendNames = {};
    coveredKeys = {};
    for k = 1:numel(keys)
        if ~ismember(keys{k},C.Properties.VariableNames)
            continue;
        end
        coveredKeys{end+1} = keys{k}; %#ok<AGROW>
        v = C.(keys{k})/1000;
        if all(abs(v) < 1e-12)
            continue;
        end
        firstPatch = gobjects(1);
        for j = 1:3
            if v(j) >= 0
                lower = positive(j);
                upper = lower + v(j);
                positive(j) = upper;
            else
                upper = negative(j);
                lower = upper + v(j);
                negative(j) = lower;
            end
            p = patch(ax,x(j)+[-0.29 0.29 0.29 -0.29], ...
                [lower lower upper upper],colors(k,:), ...
                'EdgeColor',[0.20 0.20 0.20],'LineWidth',0.4);
            if j == 1
                firstPatch = p;
            else
                p.HandleVisibility = 'off';
            end
        end
        handles(end+1) = firstPatch; %#ok<AGROW>
        legendNames{end+1} = labels{k}; %#ok<AGROW>
    end
    % An unrecognized nonzero component must not disappear from a stack.
    cols = C.Properties.VariableNames;
    extra = cols(endsWith(cols,'_increment_MW') & ~strcmp(cols,'total_increment_MW') ...
        & ~ismember(cols,coveredKeys));
    for k = 1:numel(extra)
        if any(abs(C.(extra{k})) > 1e-6)
            error('ForecastFigures:UnknownComponent', ...
                'Component %s needs an explicit legend entry before plotting.',extra{k});
        end
    end
    assertNear(1000*(positive+negative),C.total_increment_MW, ...
        'The visible component stack does not equal the signed total.');
    net = plot(ax,x,C.total_increment_MW/1000,'d','LineStyle','none', ...
        'Color',cfg.black,'MarkerFaceColor','w','MarkerSize',6.5,'LineWidth',1.5);
    yline(ax,0,'-','Color',cfg.black,'LineWidth',0.85,'HandleVisibility','off');
    xlim(ax,[0.4 3.6]);
    xticks(ax,x);
    xticklabels(ax,cellstr(string(selectedYears)));
    setPaddedY(ax,[positive;negative]);
    ax.XGrid = 'off';
    labelAxes(ax,cfg,'Calendar year',{'Increment at the annual','critical month [GW]'}, ...
        [name ': physical contributions in the turnover scenario']);
    legendBelow(ax,[net,handles],[{'Net structural increment'},legendNames],3,cfg);
end

function [cases,labels] = sensitivityNames()
    cases = {'MILESTONES','TURNOVER','FAST_RENEWAL','MANAGED_CHARGING', ...
        'HIGH_COINCIDENCE','EV_LOW_ENERGY','EV_HIGH_ENERGY','HEAT_FOSSIL_ONLY', ...
        'HEAT_HALF_JOULE','COLD_BACKUP','BASELINE_OVERLAP50','PROJECTS_DELAY3'};
    labels = {'Vehicle stocks held after 2030','Reference turnover', ...
        'Faster BEV replacement','Lower EV coincidence','Higher EV coincidence', ...
        'Lower vehicle electricity use','Higher vehicle electricity use', ...
        'No resistance-heating replacement','50% resistance-heating replacement', ...
        'Cold-weather COP and backup','50% additionality','Delayed projects'};
end

function fig = plotSensitivity(D,name,cfg)
    [fig,ax] = makeAxes(cfg,'sensitivity');
    S = D.sensitivity;
    reference = S.P50_MW(S.scenario == "TURNOVER")/1000;
    if numel(reference) ~= 1
        error('ForecastFigures:ReferenceCase', 'The sensitivity table needs exactly one TURNOVER case.');
    end
    values = S.P50_MW/1000;
    xline(ax,reference,'--','Color',cfg.gray,'LineWidth',1.1,'HandleVisibility','off');
    for k = 1:height(S)
        if values(k) < reference-1e-10
            color = cfg.blue;
        elseif values(k) > reference+1e-10
            color = cfg.orange;
        else
            color = cfg.black;
        end
        plot(ax,[reference values(k)],[k k],'-','Color',color,'LineWidth',2.0);
        plot(ax,values(k),k,'o','Color',color,'MarkerFaceColor',color, ...
            'MarkerSize',5,'LineWidth',0.8);
    end
    ax.YDir = 'reverse';
    ax.YGrid = 'off';
    ax.FontSize = 10.5;
    yticks(ax,1:height(S));
    yticklabels(ax,cellstr(S.description));
    ylim(ax,[0.5 height(S)+0.5]);
    span = max(values)-min(values);
    pad = max(0.5,0.07*span);
    xlim(ax,[min(values)-pad,max(values)+pad]);
    labelAxes(ax,cfg,sprintf('Conditional annual P50 in %d [GW]',D.terminalYear),'', ...
        [name ': structural sensitivities, not probability intervals']);
end

function setPaddedY(ax,values)
    bottom = min(values(:));
    top = max(values(:));
    span = top-bottom;
    if span <= 0
        span = max(abs(top)*0.1,1);
    end
    ylim(ax,[bottom-0.055*span,top+0.08*span]);
end

function exportOneFigure(fig,folder,name,cfg)
    objects = findall(fig,'-property','FontName');
    set(objects,'FontName',cfg.fontName);
    textObjects = findall(fig,'Type','text');
    set(textObjects,'Interpreter','none');
    drawnow;
    ax = findall(fig,'Type','axes');
    if numel(ax) ~= 1
        error('ForecastFigures:AxesCount', 'Each exported manuscript figure must have exactly one axes.');
    end
    ax.Units = 'normalized';
    ax.LooseInset = max(ax.TightInset,[0.025 0.025 0.025 0.025]);
    drawnow;
    % Include axes labels and the outside legend in each export.
    pngFile = fullfile(folder,[name '.png']);
    pdfFile = fullfile(folder,[name '.pdf']);
    exportgraphics(ax,pngFile,'Resolution',cfg.dpi,'BackgroundColor','white');
    exportgraphics(ax,pdfFile,'ContentType','vector','BackgroundColor','white');
    if cfg.saveEditableFIG
        savefig(fig,fullfile(folder,[name '.fig']));
    end
    info = imfinfo(pngFile);
    if info.Width < 1000 || info.Height < 600
        error('ForecastFigures:SmallImage', 'Unexpectedly small PNG export: %s',pngFile);
    end
    requireFile(pdfFile);
    fid = fopen(pdfFile,'r');
    if fid < 0
        error('ForecastFigures:PDFRead', 'Cannot inspect the exported PDF.');
    end
    magic = char(fread(fid,5,'*char')');
    fclose(fid);
    if ~strcmp(magic,'%PDF-')
        error('ForecastFigures:InvalidPDF', 'The exported file is not a PDF: %s',pdfFile);
    end
end

function writeJSON(file,data)
    fid = fopen(file,'w','n','UTF-8');
    if fid < 0
        error('ForecastFigures:OutputWrite', 'Cannot write %s.',file);
    end
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid,'%s\n',jsonencode(data));
end

function removeTemporaryFolder(folder)
    if isfolder(folder)
        try
            rmdir(folder,'s');
        catch
            warning('ForecastFigures:TemporaryFolder', ...
                'The temporary extraction folder could not be removed: %s',folder);
        end
    end
end

function [H, source] = observedHistory(country, root, B)
% Prefer the input snapshot saved with the results.
    here = fileparts(mfilename('fullpath'));
    packageRoot = fileparts(here);
    candidates = { ...
        fullfile(root,country,'observed_monthly_peak_MW.csv'), ...
        fullfile(root,'statistical_recalculation',country,'observed_monthly_peak_MW.csv'), ...
        fullfile(root,'observed_inputs',[upper(country) '_monthly_peak_MW.txt']), ...
        fullfile(root,'observed_inputs',[upper(country) '_monthly_peak_MW.csv']), ...
        fullfile(root,'article','figure_data',[country '_history.txt']), ...
        fullfile(packageRoot,'data','statistical',[upper(country) '_monthly_peak_MW.txt']), ...
        fullfile(packageRoot,'data','statistical',[upper(country) '_monthly_peak_MW.csv'])};
    source = '';
    for k = 1:numel(candidates)
        if isfile(candidates{k})
            source = candidates{k};
            break;
        end
    end
    if isempty(source)
        error('ForecastFigures:MissingHistory', ...
            ['No observed monthly history was found for %s. Keep this file in ', ...
             'the package matlab/ folder, or use a results ZIP containing ', ...
             '%s/observed_monthly_peak_MW.csv. See README.txt.'],country,country);
    end
    H = readObservedFile(source,country);
    H = sortrows(H,'ds');
    if height(H) < 120 || any(H.y_mw <= 0) || any(~isfinite(H.y_mw)) || ...
            any(isnat(H.ds)) || numel(unique(H.ds)) ~= height(H) || ...
            any(day(H.ds) ~= 1) || any(H.ds(2:end) ~= H.ds(1:end-1)+calmonths(1))
        error('ForecastFigures:HistoryData', ...
            'Expected at least 120 consecutive, positive monthly MW observations: %s',source);
    end
    targetDates = dateColumn(B,'ds');
    observed = numericColumn(B,'y_true');
    [found,index] = ismember(targetDates,H.ds);
    if ~all(found)
        error('ForecastFigures:HistoryDates','Benchmark target dates lie outside the input history.');
    end
    assertNear(H.y_mw(index),observed,'Observed history and benchmark targets differ.');
end

function H = readObservedFile(source,country)
% Read ds and y_mw without rescaling either country's observations.
    fid = fopen(source,'r','n','UTF-8');
    if fid < 0
        error('ForecastFigures:HistoryRead','Cannot read %s.',source);
    end
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    header = '';
    while true
        line = fgetl(fid);
        if ~ischar(line)
            error('ForecastFigures:HistoryRead','No data header found in %s.',source);
        end
        line = strtrim(strrep(line,char(65279),''));
        if isempty(line)
            continue;
        end
        if startsWith(line,'#')
            token = regexp(line,'^#\s*([^=]+)=(.*)$','tokens','once');
            if ~isempty(token)
                key = lower(strtrim(token{1}));
                value = strtrim(token{2});
                if strcmp(key,'units') && ~strcmp(value,'MW')
                    error('ForecastFigures:HistoryUnits','Observed demand must be in MW: %s.',source);
                elseif strcmp(key,'country') && ~strcmpi(value,country)
                    error('ForecastFigures:HistoryCountry','Country metadata does not match %s.',country);
                elseif strcmp(key,'frequency') && ~strcmpi(value,'monthly')
                    error('ForecastFigures:HistoryFrequency','Expected monthly observations.');
                elseif strcmp(key,'variable') && ~strcmp(value,'monthly_peak_demand')
                    error('ForecastFigures:HistoryVariable','Expected monthly_peak_demand.');
                end
            end
        else
            header = line;
            break;
        end
    end
    [~,~,extension] = fileparts(source);
    if strcmpi(extension,'.csv')
        delimiter = ',';
    else
        delimiter = char(9);
    end
    names = strtrim(strsplit(header,delimiter));
    if ~isequal(names,{'ds','y_mw'})
        error('ForecastFigures:HistoryColumns','Expected two columns ds and y_mw in %s.',source);
    end
    rows = textscan(fid,'%s%f','Delimiter',delimiter,'CommentStyle','#', ...
                    'ReturnOnError',false);
    if isempty(rows{1}) || numel(rows{1}) ~= numel(rows{2})
        error('ForecastFigures:HistoryRead','Invalid date/value rows in %s.',source);
    end
    dates = datetime(rows{1},'InputFormat','yyyy-MM-dd');
    dates.Format = 'yyyy-MM-dd';
    H = table(dates,rows{2},'VariableNames',{'ds','y_mw'});
end
