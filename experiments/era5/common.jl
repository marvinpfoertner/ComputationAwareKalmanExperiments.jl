using ComputationAwareKalman
using ComputationAwareKalmanExperiments
using Dates
using JLD2
using LoggingExtras
using Statistics
using TerminalLoggers
using ProgressLogging

Logging.global_logger(
    TeeLogger(
        ActiveFilteredLogger(
            log_args -> typeof(log_args.message) == ProgressLogging.ProgressString,
            TerminalLogger(),
        ),
        MinLevelLogger(TerminalLogger(), Logging.Warn),
    ),
)

include("common/data.jl")
include("common/model.jl")
include("common/config.jl")
include("common/metrics.jl")
