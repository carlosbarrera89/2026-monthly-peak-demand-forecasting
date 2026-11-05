MONTHLY PEAK-DEMAND FORECASTING

Code and computational material for:

Monthly Peak-Demand Forecasting and Conditional Long-Horizon Scenarios
with a Tri-Model Ensemble

The repository contains the statistical forecasting workflow, the physical
scenario accounting, archived model outputs, input snapshots, numerical
checks, the extended analysis, and the scripts used to generate the figures.
The main application covers Spain and France with monthly peak-demand data
from 2015 to 2025 and conditional scenarios for 2026 to 2045.


QUICK START

Extract the repository into a writable folder. From that folder, install the
base dependencies:

    python -m pip install -r requirements.txt

Run the archived experiment:

    python run_forecasting_final.py

The same file can be opened in Spyder and executed with F5. The default
settings reproduce the supplied Spain and France calculation:

    MODE = 'REPRODUCE_RELEASE'
    COUNTRIES = ['SPAIN', 'FRANCE']
    STATISTICAL_ONLY = False

Paths are resolved from the script location, so the current working directory
does not affect the input lookup. The workflow does not download data or
install packages automatically.

A quick input and configuration check is available without running the full
calculation:

    python run_forecasting_final.py --check-only


REPOSITORY CONTENTS

    run_forecasting_final.py
        Main statistical and physical-scenario workflow.

    data/
        Monthly peak demand, exogenous-driver snapshots, structural inputs,
        source records, and the two conceptual illustrations.

    frozen_statistical/
        Archived component forecasts, simulation banks, benchmark paths,
        fitted parameter files, and probability-selection records.

    verified_outputs/
        Numerical outputs used to check the archived reproduction.

    extended_analysis/
        Additional diagnostics for training-window length, horizons beyond
        24 months, residual assumptions, alternative baselines, and the
        separation of statistical and structural variation.

    manuscript_figures/
        Figure files used in the manuscript, together with a filename map.

    matlab/
        MATLAB exporter for the numerical forecasting figures.

    provenance/
        Source identities, configurations, assumptions, environment records,
        and evidence tables.

    validation/
        Validation records for the packaged workflow and extended analysis.

    requirements.txt
        Dependencies for archived reproduction without neural retraining.

    requirements_retraining.txt
        Additional dependencies for LSTM, LightGBM, and TFT fitting.

    frozen_input_manifest.json
        SHA-256 checksums for the verified input set.

    LICENSE
        MIT License for the project source code.

    CITATION.cff
        Citation metadata for GitHub and Zenodo.


DEMAND INPUT FORMAT

Demand is read directly in MW. The program does not infer units or apply a
country-specific magnitude correction. Dates are normalized to the first day
of each month. Missing months, duplicate dates, nonfinite values, and
nonpositive demand values stop the run.

A TXT input contains five metadata lines followed by tab-separated columns:

    # country=FRANCE
    # source=RTE
    # frequency=monthly
    # variable=monthly_peak_demand
    # units=MW
    ds<TAB>y_mw
    2015-01-01<TAB>86682
    2015-02-01<TAB>91965

<TAB> means a tab character. Save the file as UTF-8 and do not add punctuation
after MW. CSV files use the same metadata, a comma-separated ds,y_mw header,
and the .csv extension.

The supplied demand files are:

    data/statistical/SPAIN_monthly_peak_MW.txt
    data/statistical/FRANCE_monthly_peak_MW.txt

Both contain 132 monthly observations from January 2015 through December
2025. Model standardization is fitted inside each training window. Forecasts
are converted back to MW before combination and evaluation. Figures display
GW only where the axis label states GW.


ARCHIVED EXPERIMENT REPRODUCTION

The default mode reuses the archived LSTM, LightGBM, and TFT paths. It then
recalculates:

    - inverse-RMSE ensemble weights;
    - chronological residual-model selection;
    - 5000 dependence-aware statistical paths;
    - signed physical contributions;
    - 23 structural configurations per country;
    - annual maxima and P5/P50/P95 values; and
    - point and probabilistic metrics.

ETS and SARIMA scores are evaluated from archived predictions in this mode;
their historical optimizers are not rerun. The calculated outputs are checked
against verified values and the run stops when a checked result exceeds its
tolerance.

A completed run is written to:

    results/run_YYYYMMDD_HHMMSS_microseconds/

Important outputs include:

    spain/ and france/
        Scenario tables, physical contributions, and benchmark results.

    observed_inputs/
        Exact demand inputs used in the run.

    statistical_recalculation/
        Recomputed weights, error parameters, and simulated paths.

    planning_tables/
        Annual peak-demand inputs for every structural configuration.

    article/figure_data/
        Numeric series used by the plotting workflow.

    article/tables/
        Checked table exports.

    execution_status.json
        Completion state and validation checks.

    REPORT.txt
        Short execution summary.

The script also creates RESULTS_TO_SHARE.zip at the repository root and writes
the latest run path to LATEST_RUN.txt. Existing run folders are never reused.

Do not regenerate the input manifest simply to accept changed observations.
A changed input set is a new experiment and should be kept separate from the
archived reproduction.


EXTENDED ANALYSIS

The extended analysis is independent of the main scenario calculation and
leaves its weights, Monte Carlo bank, and physical cases unchanged.

Run the diagnostics based on archived outputs:

    python extended_analysis/run_extended_analysis.py

The analysis covers:

    - forecasting error versus the number of supervised training targets;
    - retrospective error by horizon and horizon band;
    - conditional coverage by horizon band;
    - terminal residual-scale sensitivity without reselection;
    - post-24-month residual-dispersion alternatives;
    - the same physical ledger applied to alternative statistical baselines;
    - the conditional statistical interval versus the range across structural
      scenario P50 values; and
    - the role and future treatment of each covariate.

A completed extended run with long-horizon retraining is included in:

    extended_analysis/validated_outputs/

That run contains 17 CSV tables, 16 PNG figures, 16 PDF figures, eight training
logs, an execution record, and a summary report. Its status file records that
the main experiment was not modified.

To repeat the 36- to 54-month retrospective fits, install the additional
packages and run:

    python -m pip install -r requirements_retraining.txt
    python extended_analysis/run_extended_analysis.py --long-horizon-retraining

The longer paths use the same LSTM, LightGBM, and TFT specifications at later
historical origins. They provide observed evidence beyond 24 months, but they
do not constitute validation of a 240-month forecast.

New extended runs are written to extended_analysis/results/. The script also
creates extended_analysis/EXTENDED_ANALYSIS_TO_SHARE.zip and updates
extended_analysis/LATEST_RUN.txt.

The residual alternatives are diagnostic cases, not a new parameter search.
The structural range is reported separately from the conditional P5/P50/P95
band because the structural configurations do not have probability weights.


MANUSCRIPT FIGURES

The folder manuscript_figures/ contains the supplied manuscript figures in
PNG and PDF where available. The correspondence between printed figure
numbers and files is listed in:

    manuscript_figures/figure_filename_map.csv

The two composite diagnostic figures can be rebuilt from the extended-analysis
tables with:

    python extended_analysis/build_manuscript_figures.py

The MATLAB exporter produces the numerical forecasting figures in Times New
Roman as 600 dpi PNG and vector PDF:

    matlab/generate_forecasting_figures.m

Open the file in MATLAB and press Run, then select RESULTS_TO_SHARE.zip. It can
also be called directly:

    generate_forecasting_figures('/path/to/RESULTS_TO_SHARE.zip')
    generate_forecasting_figures('/path/to/completed_run', '/path/to/output')

MATLAB R2020b or later is recommended. No additional MATLAB toolbox is used.
Observed histories are read from the selected results; no demand history is
embedded in the MATLAB source.

Annual quantiles are read from the sample-wise annual maxima. They are not
constructed as maxima of monthly percentile curves. Signed heating terms are
kept signed in the contribution figures.


RETRAINING OR ADDING ANOTHER COUNTRY

The input format is the same for every country. A new country requires at
least 120 consecutive monthly peak observations and its own exogenous data in:

    data/statistical/exogenous_cache/exogenous_annual_worldbank.csv
    data/statistical/exogenous_cache/exogenous_monthly_eurostat.csv

The uppercase country identifier must be consistent across the demand and
exogenous files. Availability lags and history checks are applied at every
forecast origin.

Check a new statistical experiment before fitting:

    python run_forecasting_final.py --retrain --statistical-only \
        --countries ITALY --check-only

Run the fit after the inputs pass inspection:

    python run_forecasting_final.py --retrain --statistical-only \
        --countries ITALY

The statistical route fits all three learners for the selected country and
generates its own calibration, audit, terminal, and future statistical
outputs. It does not reuse the Spanish or French predictions.

The physical layer supplied here is configured for Spain and France. A new
country needs its own vehicle stocks, heating assumptions, project capacities,
and coincidence settings. No physical scenario is inferred from a country
name.


INTERPRETATION OF THE OUTPUTS

The structural configurations are conditional cases without assigned
probabilities. Their P5/P50/P95 values describe statistical variation within
a fixed configuration. They do not cover the full uncertainty associated with
technology pathways, policy, project timing, structural parameters, or future
model deterioration.

The historical evaluation is pseudo-real-time because publication-lag rules
are applied to revised data snapshots. The terminal period was examined during
development and is described as a retrospective block rather than an untouched
prospective test.

The package produces demand inputs for later generation and transmission
planning studies. It does not select investments, reserve margins, or network
reinforcements.


VALIDATION RECORD

The packaged main workflow preserves the 264 monthly demand observations and
the archived numerical experiment. The recorded release checks include input
format validation, date alignment, component accounting, manifest checks, and
numerical comparisons against verified outputs.

The included long-horizon extended run completed successfully with:

    long_horizon_retraining: true
    tables: 17
    PNG figures: 16
    PDF figures: 16
    training logs: 8

The execution record is stored at:

    extended_analysis/validated_outputs/execution_status.json

Training logs retain library warnings and execution messages so that the run
can be inspected without hiding nonfatal diagnostics. Exact neural retraining
on a different software stack is not claimed to be bitwise identical.


LICENSE AND THIRD-PARTY MATERIAL

The project source code is released under the MIT License. See LICENSE.

The MIT License applies to code written for this repository. Data obtained
from REE/ESIOS, ENTSO-E, Eurostat, the World Bank, RTE, MITECO, ANFAC, and other
cited institutions remain subject to the terms of their original providers.
The repository does not relicense third-party publications, trademarks, or
source material.

For citation metadata, see CITATION.cff. After a GitHub release is archived in
Zenodo, the generated DOI can be added to the citation record and the article's
data and code availability statement.
